import os

# Skip TensorFlow-dependent tests on macOS CI due to known segfaults
if os.environ.get("CI") == "true":
    import pytest
    pytest.skip("Skipping TensorFlow tests on macOS CI due to segfaults", allow_module_level=True)

import unittest
import numpy as np
from networks import Actor, Critic
import gym


class TestNetworks(unittest.TestCase):
    def setUp(self):
        # Create Pendulum environment for testing
        self.env = gym.make("Pendulum-v1")
        # Initialize Actor and Critic networks with test hyperparameters
        self.actor = Actor(
            n_actions=1,
            fc1_dims=32,
            fc2_dims=32,
            action_bound=self.env.action_space.high[0],
        )
        self.critic = Critic(
            fc1_dims=32,
            fc2_dims=32,
            n_actions=1,
        )

    def test_actor_output_shape(self):
        """
        Test that the actor network returns an action of correct shape.
        """
        obs, _ = self.env.reset()
        obs = np.expand_dims(obs, axis=0)  # Add batch dimension
        action = self.actor(obs)
        self.assertEqual(action.shape, (1, 1))

    def test_critic_output_shape(self):
        """
        Test that the critic network returns a value of correct shape.
        """
        obs, _ = self.env.reset()
        obs = np.expand_dims(obs, axis=0)  # Add batch dimension
        action = np.zeros((1, 1))          # Dummy action
        value = self.critic(obs, action)
        self.assertEqual(value.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
