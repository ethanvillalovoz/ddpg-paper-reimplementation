import unittest
import gym
import numpy as np
from env_wrappers import NormalizedEnv


class TestNormalizedEnv(unittest.TestCase):
    def setUp(self):
        self.env = NormalizedEnv(gym.make("Pendulum-v1"))

    def test_reset_and_step(self):
        obs, info = self.env.reset()
        self.assertTrue(np.all(obs <= 1.0) and np.all(obs >= -1.0))
        action = self.env.action_space.sample()
        new_obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(np.all(new_obs <= 1.0) and np.all(new_obs >= -1.0))


if __name__ == "__main__":
    unittest.main()
