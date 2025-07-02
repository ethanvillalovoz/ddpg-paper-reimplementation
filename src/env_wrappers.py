import gym
import numpy as np


class NormalizedEnv(gym.Wrapper):
    """
    Gym environment wrapper for normalizing observations and actions.
    """

    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info

    def step(self, action):
        action = self._normalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_obs(obs), reward, terminated, truncated, info

    def _normalize_obs(self, obs):
        # Example: scale observations to [-1, 1] if possible
        if hasattr(self.observation_space, "high") and hasattr(
            self.observation_space, "low"
        ):
            high = self.observation_space.high
            low = self.observation_space.low
            # Avoid division by zero
            scale = np.where(high - low == 0, 1, high - low)
            return 2.0 * (obs - low) / scale - 1.0
        return obs

    def _normalize_action(self, action):
        # Example: clip actions to valid range
        return np.clip(action, self.action_space.low, self.action_space.high)
