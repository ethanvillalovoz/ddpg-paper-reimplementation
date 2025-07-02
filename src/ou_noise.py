import numpy as np
from typing import Optional

class OUActionNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.
    Commonly used in DDPG for exploration in continuous action spaces.
    """
    def __init__(self, mu: np.ndarray, sigma: float = 0.2, theta: float = 0.15, dt: float = 1e-2, x0: Optional[np.ndarray] = None):
        """
        Initialize the noise process.
        Args:
            mu (np.ndarray): Mean of the noise.
            sigma (float): Volatility parameter.
            theta (float): Speed of mean reversion.
            dt (float): Time step.
            x0 (np.ndarray or None): Initial state.
        """
        self.mu = mu                  # Mean of the noise process
        self.sigma = sigma            # Volatility parameter
        self.theta = theta            # Speed of mean reversion
        self.dt = dt                  # Time step
        self.x0 = x0                  # Initial state
        self.reset()                  # Initialize the previous state

    def __call__(self) -> np.ndarray:
        """
        Generate the next noise value using the OU process.
        Returns:
            np.ndarray: The next noise value.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x  # Update previous state
        return x

    def reset(self) -> None:
        """
        Reset the process to the initial state or zeros if not provided.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
