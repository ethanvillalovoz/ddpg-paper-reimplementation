# Need a replay buffer class
# Need a class for a target Q network (function of state, action)
# We will use batch normalization
# The policy is deterministic, how to handle explore exploit?
# Deterministic policy means outputs the actual action insteat of a probability distribution
# Will need a way to bound the actions to the environment limits
# We gave two actor abd two critic networks, a target for each
# Updates are soft, according to theta_prime = tau * theta + (1 - tau) * theta_prime, with tau << 1
# The targert actor is just the evaluation actor plus some noise process
# They used Ornstein-Uhlenbeck noise for exploration

# Imports
import os
import numpy as np
import tensorflow as tf

class OUActionNoise(object):
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.
    Used in DDPG for exploration in continuous action spaces.
    Inherits from 'object' for compatibility with Python 2/3 style classes.
    """
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        """
        Initialize the OUActionNoise process.

        Args:
            mu (np.ndarray): The mean value (center) of the noise process.
            sigma (float or np.ndarray): The volatility (scale) of the noise.
            theta (float): The rate at which the noise reverts to the mean.
            dt (float): The time step for discretization.
            x0 (np.ndarray or None): Optional initial state for the process.
        """
        self.mu = mu                  # Mean of the noise process
        self.sigma = sigma            # Volatility parameter
        self.theta = theta            # Speed of mean reversion
        self.dt = dt                  # Time step
        self.x0 = x0                  # Initial state
        self.reset()                  # Initialize the previous state

    def __call__(self):
        """
        Generate the next noise value using the OU process.

        Returns:
            np.ndarray: The next noise value, updated from the previous state.
        """
        # Compute next value based on previous value, mean, and random noise
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x  # Update previous state
        return x
    
    def reset(self):
        """
        Reset the process to the initial state or zeros if not provided.
        """
        # Set previous state to initial value or zeros
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=npfloat32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] =  1 - int(done)
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal  