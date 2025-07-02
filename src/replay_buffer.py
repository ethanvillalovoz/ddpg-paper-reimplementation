import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for DDPG.
    Stores transitions and allows random sampling for training.
    """
    def __init__(self, max_size, input_shape, n_actions):
        # Maximum number of transitions to store
        self.mem_size = max_size
        # Counter to keep track of the number of transitions added
        self.mem_cntr = 0
        # Memory arrays for states, new states, actions, rewards, and terminal flags
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        """
        Store a new experience in the buffer, overwriting the oldest if full.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            new_state (np.ndarray): Next state after action.
            done (bool): Whether the episode ended after this transition.
        """
        index = self.mem_cntr % self.mem_size  # Circular buffer index
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)  # 0 if done, 1 otherwise
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            Tuple of (states, actions, rewards, new_states, terminals)
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # Only sample from filled part
        batch = np.random.choice(max_mem, batch_size)  # Random indices
        return (self.state_memory[batch],
                self.action_memory[batch],
                self.reward_memory[batch],
                self.new_state_memory[batch],
                self.terminal_memory[batch])
