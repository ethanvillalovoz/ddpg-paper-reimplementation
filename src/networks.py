"""
networks.py

Defines the Actor and Critic neural network classes for DDPG.
"""

import os
import numpy as np
import tensorflow as tf

class ActorNetwork(tf.keras.Model):
    """
    Actor network for DDPG using tf.keras.Model.
    Outputs deterministic actions given states.
    """
    def __init__(self, n_actions, fc1_dims, fc2_dims, input_dims, action_bound):
        super(ActorNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation='tanh')
        self.action_bound = action_bound

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        # Scale output to action bounds
        return mu * self.action_bound

class Critic(tf.keras.Model):
    """
    Critic network for DDPG using tf.keras.Model.
    Outputs Q-values given states and actions.
    """
    def __init__(self, fc1_dims, fc2_dims, input_dims, n_actions):
        super(Critic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.q(x)
        return q
