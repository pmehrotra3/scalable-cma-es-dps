# train_trpo.py
# Trains TRPO (SB3 Contrib) on a Gymnasium environment and logs per-episode returns/lengths
# and per-episode system metrics (wall time, CPU time, RAM, CPU%) to CSV files.
#
# Usage: python train_trpo.py <env_name> [--total-timesteps N] [--render]
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com
#   ChatGPT (OpenAI)     — https://openai.com
#   Gemini  (Google)     — https://deepmind.google

import argparse, os, csv, time, psutil, threading
import gymnasium as gym
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import BaseCallback


# -----------------------------------------------------------------------------
# EpisodeLoggerCallback
# -----------------------------------------------------------------------------

class EpisodeLoggerCallback(BaseCallback):
    """
    SB3 callback that accumulates per-episode returns and system metrics,
    then writes episode_log.csv and system_log.csv to out_dir at training end.
    RAM is sampled every 5 seconds by a background daemon thread and averaged per episode.
    """

    def __init__(self, out_dir: str, run: int):
        super().__init__()

        # Within-episode accumulators — reset to zero after every episode end.
        self.current_episode_reward: float = 0.0
        self.current_episode_length: int   = 0

        # Episode log lists — index i = episode i.
        self.episode_returns: list = []
        self.episode_lengths: list = []

        # System metric lists — index i = snapshot at end of episode i.
        self.sys_wall_times: list = []
        self.sys_cpu_times:  list = []
        self.sys_ram_mb:     list = []
        self.sys_cpu_pct:    list = []

        # Timing handles — set in _on_training_start.
        self.start_wall = None
        self.start_cpu  = None
        self.process    = None

        # RAM daemon state.
        self._ram_sum    = 0.0
        self._ram_count  = 0
        self._ram_lock   = threading.Lock()
        self._stop_event = threading.Event()

        self.out_dir: str = out_dir
        self.run: int = run

    def _ram_sampler_loop(self) -> None:
        """Background daemon: samples RSS every 5 s and adds to the running sum."""
        while not self._stop_event.wait(5.0):
            sample = self.process.memory_info().rss / 1024 / 1024
            with self._ram_lock:
                self._ram_sum   += sample
                self._ram_count += 1

    def _on_training_start(self) -> None:
        """Record start times, initialise psutil handle, and launch the RAM sampler thread."""
        self.start_wall = time.time()
        self.start_cpu  = time.process_time()
        self.process    = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None)

        self._sampler_thread = threading.Thread(target=self._ram_sampler_loop, daemon=True)
        self._sampler_thread.start()

    def _on_step(self) -> bool:
        """
        Called after every env.step(). Accumulates reward and length each step.
        On episode end, appends to log lists, snapshots system metrics, resets accumulators.
        """
        dones   = self.locals.get("dones")
        rewards = self.locals.get("rewards")

        self.current_episode_reward += rewards[0]
        self.current_episode_length += 1

        if dones[0]:

            # --- episode log ---
            self.episode_returns.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # --- system metrics ---
            self.sys_wall_times.append(time.time() - self.start_wall)
            self.sys_cpu_times.append(time.process_time() - self.start_cpu)
            self.sys_cpu_pct.append(self.process.cpu_percent(interval=None))

            # Final snapshot + average with daemon samples, then reset for next episode.
            final = self.process.memory_info().rss / 1024 / 1024
            with self._ram_lock:
                self._ram_sum   += final
                self._ram_count += 1
                avg_ram          = self._ram_sum / self._ram_count
                self._ram_sum    = 0.0
                self._ram_count  = 0
            self.sys_ram_mb.append(avg_ram)

            # --- reset for next episode ---
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        """Stop the RAM sampler thread, then write all accumulated data to CSV files."""
        self._stop_event.set()
        self._sampler_thread.join()

        # --- episode_log.csv ---
        episode_csv = os.path.join(self.out_dir, f"episode_log_run_{self.run}.csv")
        with open(episode_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "reward", "length"])
            timestep = 0
            for i, (ret, length) in enumerate(zip(self.episode_returns, self.episode_lengths)):
                timestep += length
                writer.writerow([i + 1, timestep, ret, length])

        # --- system_log.csv ---
        system_csv = os.path.join(self.out_dir, f"system_log_run_{self.run}.csv")
        with open(system_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "wall_time_s", "cpu_time_s", "ram_mb", "cpu_pct"])
            for i, (wall, cpu, ram, pct) in enumerate(zip(self.sys_wall_times, self.sys_cpu_times, self.sys_ram_mb, self.sys_cpu_pct)):
                writer.writerow([i + 1, wall, cpu, ram, pct])

        print(f"Saved episode log : {episode_csv}")
        print(f"Saved system log  : {system_csv}")

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():

    # --- command-line arguments ---
    parser = argparse.ArgumentParser(description="Train TRPO with full episode and system logging.")
    parser.add_argument("env_name",          type=str,             help="Gymnasium env ID, e.g. CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total env steps to train for")
    parser.add_argument("--render",          action="store_true",  help="Render the environment during training")
    parser.add_argument("--num-runs",        type=int, default=1,  help="Number of runs to average over (default: 1)")
    args = parser.parse_args()

    # --- environment ---
    render_mode = "human" if args.render else None
    env = gym.make(args.env_name, render_mode=render_mode)

    # --- output directory ---
    out_dir = os.path.join("..", "output", args.env_name, "TRPO")
    os.makedirs(out_dir, exist_ok=True)  # creates full path, no error if already exists

    # --- create model and train ---
    print("Training started ...")
    
    for run in range(1, args.num_runs + 1):
        print(f"Run {run}/{args.num_runs} ...")
        model = TRPO(policy="MlpPolicy",  # standard feedforward network for vector observations
                    env=env,
                    verbose=0,           # silence SB3's own built-in output
                    )
        logger = EpisodeLoggerCallback(out_dir, run)
        model.learn(total_timesteps=args.total_timesteps, callback=logger)
        
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()