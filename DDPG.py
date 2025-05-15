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
from tensorflow.keras.initializers import RandomUniform

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

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class ActorNetwork(object):
    def __init__(self, lr, n_actions, input_dims, name, sess, fcl_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.names = name
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
        self.checpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, [None, *self.input_dims], name='input')
            self.action_gradient = tf.placeholder(tf.float32, [None, self.n_actions], name='action_gradient')

            f1 = 1 / np.sqrt(self.fcl_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1dims, kernal_initializer=random_uniform(-f1, f1), bias_initializer=random_uniform(-f1, f1), name='dense1')

            batch1 = tf.layers.batch_normalization(dense1)
            activation1 = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(activation1, units=self.fc2_dims, kernal_initializer=random_uniform(-f2, f2), bias_initializer=random_uniform(-f2, f2), name='dense2')
            batch2 = tf.layers.batch_normalization(dense2)
            activation2 = tf.nn.relu(batch2)

            f3 = 0.003

            mu = tf.layers.dense(activation2, units=self.n_actions, activation='tanh', kernal_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3), name='mu')

            self.mu = tf.multiply(mu, self.action_bound, name='scaled_mu')

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})
    
    def train(self, inputs, gradients):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action_gradient: gradients})
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

class Critic(object):
    def __init__(self, lr, n_actions, input_dims, name, sess, fcl_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.names = name
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
        self.checpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.actor_gradients = tf.gradients(self.q, self.action, name='actor_gradients')

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, [None, *self.input_dims], name='input')
            self.action = tf.placeholder(tf.float32, [None, self.n_actions], name='action')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='q_target')

            f1 = 1 / np.sqrt(self.fcl_dims)
            dense1 = tf.layers.dense(self.input, units=self.fcl_dims, kernal_initializer=random_uniform(-f1, f1), bias_initializer=random_uniform(-f1, f1), name='dense1')
            batch1 = tf.layers.batch_normalization(dense1)
            activation1 = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(activation1, units=self.fc2_dims, kernal_initializer=random_uniform(-f2, f2), bias_initializer=random_uniform(-f2, f2), name='dense2')
            batch2 = tf.layers.batch_normalization(dense2)

            action_in = tf.layers.dense(self.action, units=self.fc2_dims, activation='relu')

            state_action = tf.add(batch2, action_in)
            state_action = tf.nn.relu(state_action)

            f3 = 0.003
            self.q = tf.layers.dense(state_action, units=1, kernal_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3), kernal_regularizer=tf.keras.regularizers.l2(0.01), name='q')
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.action: actions})
    def train(self, inputs, actions, target):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action: actions, self.q_target: target})
    
    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.actor_gradients, feed_dict={self.input: inputs, self.action: actions})
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)