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

class OUActionNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.
    Commonly used in DDPG for exploration in continuous action spaces.
    """
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        """
        Initialize the noise process.
        Args:
            mu (np.ndarray): Mean of the noise.
            sigma (float or np.ndarray): Volatility parameter.
            theta (float): Speed of mean reversion.
            dt (float): Time step.
            x0 (np.ndarray or None): Initial state.
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Generate the next noise value using the OU process.
        Returns:
            np.ndarray: The next noise value.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        """
        Reset the process to the initial state or zeros.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
