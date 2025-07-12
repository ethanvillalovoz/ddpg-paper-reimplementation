import sys
import os
if os.environ.get("CI") == "true":
    import pytest
    pytest.skip("Skipping TensorFlow tests on macOS CI due to segfaults", allow_module_level=True)

import unittest
from agent import Agent
import gym
from env_wrappers import NormalizedEnv


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.env = NormalizedEnv(gym.make("Pendulum-v1"))
        self.agent = Agent(
            alpha=0.0001,
            beta=0.001,
            input_dims=[3],
            tau=0.001,
            env=self.env,
            batch_size=8,
            layer1_size=32,
            layer2_size=32,
            n_actions=1,
            gamma=0.99,
            max_size=100,
        )

    def test_choose_action(self):
        obs, _ = self.env.reset()
        action = self.agent.choose_action(obs)
        self.assertEqual(action.shape, (1,))

    def test_remember_and_learn(self):
        obs, _ = self.env.reset()
        action = self.agent.choose_action(obs)
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.agent.remember(obs, action, reward, new_obs, int(done))
        self.agent.learn()


if __name__ == "__main__":
    unittest.main()
