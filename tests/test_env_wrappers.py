import unittest
import gym
import numpy as np
from env_wrappers import NormalizedEnv


class TestNormalizedEnv(unittest.TestCase):
    def setUp(self):
        # Create a normalized Pendulum environment for testing
        self.env = NormalizedEnv(gym.make("Pendulum-v1"))

    def test_reset_and_step(self):
        """
        Test that observations from reset and step are normalized to [-1, 1].
        """
        obs, info = self.env.reset()
        # Check that initial observation is within normalized bounds
        self.assertTrue(np.all(obs <= 1.0) and np.all(obs >= -1.0))
        # Sample a random action and step the environment
        action = self.env.action_space.sample()
        new_obs, reward, terminated, truncated, info = self.env.step(action)
        # Check that new observation is also within normalized bounds
        self.assertTrue(np.all(new_obs <= 1.0) and np.all(new_obs >= -1.0))


if __name__ == "__main__":
    unittest.main()
