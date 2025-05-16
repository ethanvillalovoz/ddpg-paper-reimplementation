"""
noise.py

Contains the Ornstein-Uhlenbeck process for temporally correlated exploration noise,
as used in the DDPG algorithm.
"""

import numpy as np

class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.

    Args:
        mu (np.ndarray): The mean of the noise process.
        sigma (float): The volatility parameter.
        theta (float): The speed of mean reversion.
        dt (float): The time step.
        x0 (np.ndarray or None): Initial state.
    """
    def __init__(self, mu: np.ndarray, sigma: float = 0.2, theta: float = 0.15, dt: float = 1e-2, x0: np.ndarray = None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self) -> None:
        """Reset the process to the initial state."""
        if self.x0 is None:
            self.x0 = np.zeros_like(self.mu)
        self.x_prev = self.x0

    def __call__(self) -> np.ndarray:
        """
        Generate the next noise value.

        Returns:
            np.ndarray: The next noise value.
        """
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x
