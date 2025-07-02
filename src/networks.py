import tensorflow as tf
from typing import Any


class Actor(tf.keras.Model):
    """
    Actor network for DDPG.
    Responsible for learning the deterministic policy (mapping states to actions).
    Inherits from tf.keras.Model for easy integration with TensorFlow 2.x training.
    """

    def __init__(self, n_actions: int, fc1_dims: int, fc2_dims: int, action_bound: Any):
        super(Actor, self).__init__()
        # First fully connected layer
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        # Second fully connected layer
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        # Output layer with tanh activation to bound actions between -1 and 1
        self.out = tf.keras.layers.Dense(n_actions, activation="tanh")
        # Action bound for scaling output actions to environment limits
        self.action_bound = action_bound

    def call(self, state: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for the actor network.
        Args:
            state (tf.Tensor): Input state(s).
        Returns:
            tf.Tensor: Scaled action(s).
        """
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.out(x)
        return x * self.action_bound


class Critic(tf.keras.Model):
    """
    Critic network for DDPG.
    Responsible for estimating the Q-value (expected return) for state-action pairs.
    Inherits from tf.keras.Model for easy integration with TensorFlow 2.x training.
    """

    def __init__(self, fc1_dims: int, fc2_dims: int, n_actions: int):
        super(Critic, self).__init__()
        # First fully connected layer for state input
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        # Second fully connected layer for state input
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        # Output layer for Q-value
        self.q = tf.keras.layers.Dense(1, activation=None)
        # Separate fully connected layer for action input
        self.action_fc = tf.keras.layers.Dense(fc2_dims, activation="relu")

    def call(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for the critic network.
        Args:
            state (tf.Tensor): Input state(s).
            action (tf.Tensor): Input action(s).
        Returns:
            tf.Tensor: Estimated Q-value(s).
        """
        x = self.fc1(state)
        x = self.fc2(x)
        a = self.action_fc(action)
        x = tf.keras.layers.Add()([x, a])  # Combine state and action pathways
        x = tf.nn.relu(x)
        q = self.q(x)
        return q
