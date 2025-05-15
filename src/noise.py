# Need a replay buffer class
# Need a class for a target Q network (function of state and action)
# We will use batch normalization
# The policy is deterministic, how to handle explore exploitation?
# Deterministic policy gradient means outputs the actual action instead of a probability distribution
# Will need a way to bound the action to the environment limits
# We have two actors and two critics networks, a target for each
# Updates are done in a soft way, according to theta_prime = tau * theta + (1-tau) * theta_prime, with tau << 1
# The target actor is just the evaluated actor plus noise
# They used Ornstein-Uhlenbeck process for exploration noise -> will need a class for the noise

import os
import numpy as np
np.bool8 = bool
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.initializers import RandomUniform

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        if self.x0 is None:
            self.x0 = np.zeros_like(self.mu)
        self.x_prev = self.x0

    def __call__(self):
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x
