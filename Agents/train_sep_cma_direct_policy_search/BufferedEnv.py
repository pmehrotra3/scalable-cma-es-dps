# BufferedEnv.py
# Lightweight Gymnasium env wrapper that reuses observation arrays across env steps.
# Used across all CMA-ES variants for codebase consistency.
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

import numpy as np


# =============================================================================
# BufferedEnv
# =============================================================================

class BufferedEnv:
    """
    Wraps a Gymnasium env so that observation arrays are reused across calls
    to reset() and step(), rather than the underlying env returning a freshly
    allocated array each time. The wrapper presents the same interface as the
    underlying env (observation_space, action_space, reset, step, close).
    """

    def __init__(self, env):
        self.env               = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self._obs_buf          = np.zeros(env.observation_space.shape, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        np.copyto(self._obs_buf, obs)
        return self._obs_buf, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        np.copyto(self._obs_buf, obs)
        return self._obs_buf, reward, terminated, truncated, info

    def close(self):
        self.env.close()