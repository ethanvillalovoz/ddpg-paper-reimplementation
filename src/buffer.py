"""
buffer.py

Defines the ReplayBuffer class for storing and sampling experience tuples.
"""

import numpy as np

class ReplayBuffer:
    """
    Experience Replay Buffer for DDPG agent.
    Stores transitions and allows random sampling for training.
    """
    def __init__(self, max_size: int, input_shape: tuple, n_actions: int):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum number of transitions to store.
            input_shape (tuple): Shape of the state space.
            n_actions (int): Number of action dimensions.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(
        self, state: np.ndarray, action: np.ndarray, reward: float, state_: np.ndarray, done: bool
    ) -> None:
        """
        Store a single transition in the buffer.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            Tuple of (states, actions, rewards, new_states, terminals)
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, new_states, terminal
