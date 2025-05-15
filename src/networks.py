"""
networks.py

Defines the Actor and Critic neural network classes for DDPG.
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.initializers import RandomUniform

class ActorNetwork:
    """
    Actor network for DDPG.
    Outputs deterministic actions given states.
    """
    def __init__(
        self, lr: float, n_actions: int, input_dims: list, name: str, sess,
        fcl_dims: int, fc2_dims: int, action_bound: float, batch_size: int = 64,
        chkpt_dir: str = 'tmp/ddpg'
    ):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.fcl_dims = fcl_dims
        self.fc2_dims = fc2_dims
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.sess = sess
        self.checkpoint_dir = chkpt_dir

        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        self.actor_gradients = [tf.div(x, self.batch_size) for x in self.unnormalized_actor_gradients]
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self) -> None:
        """Build the actor network graph."""
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, [None, *self.input_dims], name='input')
            self.action_gradient = tf.placeholder(tf.float32, [None, self.n_actions], name='action_gradient')

            f1 = 1 / np.sqrt(self.fcl_dims)
            dense1 = tf.keras.layers.Dense(
                self.fcl_dims,
                kernel_initializer=RandomUniform(-f1, f1),
                bias_initializer=RandomUniform(-f1, f1),
                name='dense1'
            )(self.input)
            batch1 = tf.keras.layers.BatchNormalization()(dense1)
            activation1 = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.keras.layers.Dense(
                self.fc2_dims,
                kernel_initializer=RandomUniform(-f2, f2),
                bias_initializer=RandomUniform(-f2, f2),
                name='dense2'
            )(activation1)
            batch2 = tf.keras.layers.BatchNormalization()(dense2)
            activation2 = tf.nn.relu(batch2)

            f3 = 0.003
            mu = tf.keras.layers.Dense(
                self.n_actions,
                activation='tanh',
                kernel_initializer=RandomUniform(-f3, f3),
                bias_initializer=RandomUniform(-f3, f3),
                name='mu'
            )(activation2)

            self.mu = tf.multiply(mu, self.action_bound, name='scaled_mu')

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict actions given states."""
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs: np.ndarray, gradients: np.ndarray) -> None:
        """Train the actor network."""
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action_gradient: gradients})

    def save_checkpoint(self) -> None:
        """Save network weights."""
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self) -> None:
        """Load network weights."""
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)


class Critic:
    """
    Critic network for DDPG.
    Outputs Q-values given states and actions.
    """
    def __init__(
        self, lr: float, n_actions: int, input_dims: list, name: str, sess,
        fcl_dims: int, fc2_dims: int, batch_size: int = 64, chkpt_dir: str = 'tmp/ddpg'
    ):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.fcl_dims = fcl_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.sess = sess
        self.checkpoint_dir = chkpt_dir

        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.actor_gradients = tf.gradients(self.q, self.action, name='actor_gradients')

    def build_network(self) -> None:
        """Build the critic network graph."""
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, [None, *self.input_dims], name='input')
            self.action = tf.placeholder(tf.float32, [None, self.n_actions], name='action')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='q_target')

            f1 = 1 / np.sqrt(self.fcl_dims)
            dense1 = tf.keras.layers.Dense(
                self.fcl_dims,
                kernel_initializer=RandomUniform(-f1, f1),
                bias_initializer=RandomUniform(-f1, f1),
                name='dense1'
            )(self.input)
            batch1 = tf.keras.layers.BatchNormalization()(dense1)
            activation1 = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.keras.layers.Dense(
                self.fc2_dims,
                kernel_initializer=RandomUniform(-f2, f2),
                bias_initializer=RandomUniform(-f2, f2),
                name='dense2'
            )(activation1)
            batch2 = tf.keras.layers.BatchNormalization()(dense2)

            action_in = tf.keras.layers.Dense(
                self.fc2_dims,
                activation='relu'
            )(self.action)

            state_action = tf.add(batch2, action_in)
            state_action = tf.nn.relu(state_action)

            f3 = 0.003
            self.q = tf.keras.layers.Dense(
                1,
                kernel_initializer=RandomUniform(-f3, f3),
                bias_initializer=RandomUniform(-f3, f3),
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                name='q'
            )(state_action)
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict Q-values given states and actions."""
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.action: actions})

    def train(self, inputs: np.ndarray, actions: np.ndarray, target: np.ndarray) -> None:
        """Train the critic network."""
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action: actions, self.q_target: target})

    def get_action_gradients(self, inputs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Get gradients of Q-values with respect to actions."""
        return self.sess.run(self.actor_gradients, feed_dict={self.input: inputs, self.action: actions})

    def save_checkpoint(self) -> None:
        """Save network weights."""
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self) -> None:
        """Load network weights."""
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)
