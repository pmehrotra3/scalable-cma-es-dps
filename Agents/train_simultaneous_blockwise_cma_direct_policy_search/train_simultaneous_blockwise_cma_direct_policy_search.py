# train_simultaneous_blockwise_cma_direct_policy_search.py
# Trains simultaneous_blockwise_cma_direct_policy_search (Custom implementation using pycma) on a Gymnasium environment and logs per-episode returns/lengths
# and per-episode system metrics (wall time, CPU time, RAM, CPU%) to CSV files.
#
# Usage: python train_simultaneous_blockwise_cma_direct_policy_search.py <env_name> [--total-timesteps N] [--num-runs N] [--block-size N]
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

import argparse, os, csv, time, psutil, threading
import gymnasium as gym
from simultaneous_blockwise_cma_direct_policy_search import simultaneous_blockwise_cma_direct_policy_search
from BaseCallback import BaseCallback
from BufferedEnv import BufferedEnv


# -----------------------------------------------------------------------------
# EpisodeLoggerCallback
# -----------------------------------------------------------------------------

class EpisodeLoggerCallback(BaseCallback):
    """
    Callback that accumulates per-episode returns and system metrics,
    then writes episode_log_run_N.csv and system_log_run_N.csv to out_dir at training end.
    RAM is sampled every 5 seconds by a background daemon thread and averaged per episode.
    """

    def __init__(self, out_dir: str, run: int):

        # Episode log lists — index i = episode i.
        self.episode_returns: list = []
        self.episode_lengths: list = []

        # System metric lists — index i = snapshot at end of episode i.
        self.sys_wall_times: list = []
        self.sys_cpu_times:  list = []
        self.sys_ram_mb:     list = []
        self.sys_cpu_pct:    list = []

        # Timing handles — set in on_training_start.
        self.start_wall = None
        self.start_cpu  = None
        self.process    = None

        # RAM daemon state.
        self._ram_sum    = 0.0
        self._ram_count  = 0
        self._ram_lock   = threading.Lock()
        self._stop_event = threading.Event()

        self.out_dir: str = out_dir
        self.run:     int = run

    def _ram_sampler_loop(self) -> None:
        """Background daemon: samples RSS every 5 s and adds to the running sum."""
        while not self._stop_event.wait(5.0):
            sample = self.process.memory_info().rss / 1024 / 1024
            with self._ram_lock:
                self._ram_sum   += sample
                self._ram_count += 1

    def on_training_start(self) -> None:
        """Record start times, initialise psutil handle, and launch the RAM sampler thread."""
        self.start_wall = time.time()
        self.start_cpu  = time.process_time()
        self.process    = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None)

        self._sampler_thread = threading.Thread(target=self._ram_sampler_loop, daemon=True)
        self._sampler_thread.start()

    def on_episode_end(self, ep_return: float, ep_length: int) -> None:
        """
        Called after every rollout. Takes a final RAM snapshot, averages with daemon samples,
        resets counters, then appends all metrics to log lists.
        """
        self.episode_returns.append(float(ep_return))
        self.episode_lengths.append(int(ep_length))

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

    def on_training_end(self) -> None:
        """Stop the RAM sampler thread, then write all accumulated data to CSV files."""
        self._stop_event.set()
        self._sampler_thread.join()

        # --- episode_log_run_N.csv ---
        episode_csv = os.path.join(self.out_dir, f"episode_log_run_{self.run}.csv")
        with open(episode_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "reward", "length"])
            timestep = 0
            for i, (ret, length) in enumerate(zip(self.episode_returns, self.episode_lengths)):
                timestep += length
                writer.writerow([i + 1, timestep, ret, length])

        # --- system_log_run_N.csv ---
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
    parser = argparse.ArgumentParser(description="Train simultaneous_blockwise_cma_direct_policy_search with full episode and system logging.")
    parser.add_argument("env_name",          type=str,                  help="Gymnasium env ID, e.g. HalfCheetah-v5")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total env steps to train for (default: 100000)")
    parser.add_argument("--num-runs",        type=int, default=1,       help="Number of runs to average over (default: 1)")
    parser.add_argument("--block-size",      type=int, default=8,       help="Number of neurons per block for blockwise CMA-ES (default: 8)")
    args = parser.parse_args()

    # --- environment (wrapped with BufferedEnv to reuse obs arrays across steps) ---
    env = BufferedEnv(gym.make(args.env_name))

    # --- output directory ---
    out_dir = os.path.join("..", "..", "output", args.env_name, f"simultaneous_blockwise_cma_direct_policy_search [BLOCK_SIZE = {args.block_size}]")
    os.makedirs(out_dir, exist_ok=True)

    # --- create model and train ---
    print("Training started ...")

    for run in range(1, args.num_runs + 1):
        print(f"Run {run}/{args.num_runs} ...")
        model = simultaneous_blockwise_cma_direct_policy_search(env=env, block_size=args.block_size)
        logger = EpisodeLoggerCallback(out_dir, run)
        model.learn(total_timesteps=args.total_timesteps, callback=logger)

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()