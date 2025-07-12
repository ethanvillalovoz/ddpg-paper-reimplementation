import unittest
import numpy as np
from replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_store_and_sample(self):
        """
        Test storing a transition and sampling from the replay buffer.
        """
        # Initialize replay buffer with small size and test shapes
        buffer = ReplayBuffer(max_size=10, input_shape=[3], n_actions=1)
        state = np.array([1, 2, 3], dtype=np.float32)        # Example state
        action = np.array([0.5], dtype=np.float32)           # Example action
        reward = 1.0                                         # Example reward
        new_state = np.array([4, 5, 6], dtype=np.float32)    # Example next state
        done = False                                         # Example done flag

        # Store the transition in the buffer
        buffer.store_transition(state, action, reward, new_state, done)
        # Sample one transition from the buffer
        states, actions, rewards, new_states, dones = buffer.sample_buffer(1)

        # Check shapes of sampled data
        self.assertEqual(states.shape, (1, 3))
        self.assertEqual(actions.shape, (1, 1))
        self.assertEqual(rewards.shape, (1,))
        self.assertEqual(new_states.shape, (1, 3))
        self.assertEqual(dones.shape, (1,))


if __name__ == "__main__":
    unittest.main()
