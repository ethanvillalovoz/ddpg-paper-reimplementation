import unittest
import numpy as np
from networks import Actor, Critic
import gym


class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Pendulum-v1")
        self.actor = Actor(
            n_actions=1,
            fc1_dims=32,
            fc2_dims=32,
            action_bound=self.env.action_space.high[0],
        )
        self.critic = Critic(fc1_dims=32, fc2_dims=32, n_actions=1)

    def test_actor_output_shape(self):
        obs, _ = self.env.reset()
        obs = np.expand_dims(obs, axis=0)
        action = self.actor(obs)
        self.assertEqual(action.shape, (1, 1))

    def test_critic_output_shape(self):
        obs, _ = self.env.reset()
        obs = np.expand_dims(obs, axis=0)
        action = np.zeros((1, 1))
        value = self.critic(obs, action)
        self.assertEqual(value.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
