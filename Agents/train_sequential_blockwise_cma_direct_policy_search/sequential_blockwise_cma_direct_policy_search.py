# sequential_blockwise_cma_direct_policy_search.py
# CMA-ES optimisers over the full parameter vector of a C++ neural network.
# One CMA-ES instance per block — each block covers a contiguous group of neurons within a layer.
# Only one block is updated per generation in a round-robin schedule. The active block samples a
# population of candidates; inactive blocks contribute their current distribution mean to the
# assembled network during fitness evaluation, giving the active block a clean credit signal.
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

import numpy as np
import cma
import nn
from BaseCallback import BaseCallback


HIDDEN_LAYERS = (64, 64)  # Fixed architecture — matches SB3 MlpPolicy defaults for fair comparison.
SIGMA         = 0.05      # CMA-ES initial step size — pycma default
BLOCK_SIZE    = 1         # Default number of neurons per block


# =============================================================================
# sequential_blockwise_cma_direct_policy_search
# =============================================================================

class sequential_blockwise_cma_direct_policy_search:
    """
    CMA-ES optimisers over the full parameter vector of a C++ neural network.
    The parameter vector is partitioned into blocks (groups of neurons within each layer).
    Each block has its own CMA-ES instance with its own covariance matrix.
    Only ONE block is active per generation, in a round-robin schedule.
    The active block samples its own population of candidates; the other blocks contribute their
    current distribution mean (es.mean) to the assembled network during fitness evaluation.
    After the active block's tell() call, the next block in the cycle becomes active.
    Interface mirrors SB3: model.learn(total_timesteps, callback=callback)
    """

    def __init__(self, env, block_size: int = BLOCK_SIZE):

        self.env        = env
        self.block_size = block_size
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])

        # Build C++ neural network with blockwise parameter partitioning.
        self.nn = nn.NeuralNetwork(obs_dim, list(HIDDEN_LAYERS), act_dim, block_size)

        # Get the initial parameter blocks — one list per block.
        # init_blocks is a list of lists: [ block_0_params, block_1_params, ... ]
        init_blocks   = self.nn.get_param()
        self.n_blocks = len(init_blocks)

        # One CMA-ES instance per block.
        
        opts = {
            "CMA_diagonal": 0,
            "verbose":      -9,
            "popsize_factor": 0.5
        }
        self.es_list = [cma.CMAEvolutionStrategy(block, SIGMA, opts)for block in init_blocks]

        # Training state.
        self.global_steps = 0
        self.best_blocks  = None
        self.best_score   = -np.inf

    def predict(self, obs):
        """Forward pass through the network."""
        out = np.asarray(self.nn.forward(np.asarray(obs, dtype=np.float64).tolist()), dtype=np.float64)
        return out

    def _episode(self, blocks, callback: BaseCallback | None):
        """
        Loads a list of blocks into the network and runs one full episode.
        Returns total episode return.
        """
        self.nn.set_param(blocks)
        obs, _ = self.env.reset()
        ep_ret, ep_len, done = 0.0, 0, False

        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done    = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
            self.global_steps += 1

        if callback is not None:
            callback.on_episode_end(ep_ret, ep_len)

        return ep_ret

    def learn(self, total_timesteps: int, callback: BaseCallback | None = None):
        """
        Runs sequential blockwise CMA-ES until total_timesteps environment steps have been taken.

        Round-robin schedule — one block active per generation:
        1. Pick the active block (round-robin through all blocks).
        2. The active block's CMA-ES asks for a population of candidate vectors.
        3. For each candidate, assemble the full network: that candidate for the active block,
            plus the best-known blocks for every inactive block (or the distribution mean
            if no best is yet known).
        4. Evaluate fitness via one episode per candidate.
        5. Tell the active block's CMA-ES the fitness scores.
        6. Move to the next block. Continue cycling until budget exhausted.
        """
        if callback is not None:
            callback.on_training_start()

        active_idx = 0  # round-robin pointer

        while self.global_steps < total_timesteps:

            # Step 1 & 2: active block asks for its population.
            active_es        = self.es_list[active_idx]
            active_solutions = active_es.ask()
            popsize          = len(active_solutions)

            losses = []

            # Snapshot the inactive blocks: use best-known if available, else distribution mean.
            if self.best_blocks is not None:
                inactive_blocks = [np.array(blk) for blk in self.best_blocks]
            else:
                inactive_blocks = [np.array(self.es_list[b].mean) for b in range(self.n_blocks)]

            # Step 3 & 4: assemble full network candidate-by-candidate and evaluate.
            for i in range(popsize):
                if self.global_steps >= total_timesteps:
                    break

                # Build the full block list: active block uses candidate i, others use best/mean.
                blocks             = list(inactive_blocks)
                blocks[active_idx] = active_solutions[i]

                score = self._episode(blocks, callback=callback)
                losses.append(-score)  # CMA-ES minimises — negate reward

                if score > self.best_score:
                    self.best_score  = score
                    self.best_blocks = [np.array(blk) for blk in blocks]

            # Step 5: tell the active block's CMA-ES the losses (only if full generation evaluated).
            if len(losses) == popsize:
                active_es.tell(active_solutions, losses)

            # Step 6: advance round-robin pointer.
            active_idx = (active_idx + 1) % self.n_blocks

        # Load best found blocks into the network.
        if self.best_blocks is not None:
            self.nn.set_param(self.best_blocks)

        if callback is not None:
            callback.on_training_end()

        return self