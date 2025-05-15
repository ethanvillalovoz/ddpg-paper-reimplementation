# Need a replay buffer class
# Need a class for a target Q network (function of state and action)
# We will use batch normalization
# The policy is deterministic, how to handle explore exploitation?
# Deterministic policy gradient means outputs the acutal action instead of a probability distribution
# Will need a way to bound the action to the environment limits
# We have two actors and two critics networks, a target for each
# Updates are done in a soft way, according to theta_prime = tau * theta + (1-tau) * theta_prime, wth tau << 1
# The target actor is just the evaluated actor plus noise
# They used Ornstein-Uhlenbeck process for exploration noise -> will need a class for the noise

import os
import numpy as np
import tensorflow as tf

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=0.2, dt=1e-2, x0=None):
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
    
    class ReplayBuffer(object):
        def __init__(self, max_size, input_shape, n_actions):
            self.mem_size = max_size
            self.mem_cntr = 0
            self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
            self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
            self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
            self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

        def store_transition(self, state, action, reward, state_, done):
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = 1 - int(done)
            self.mem_cntr += 1

        
