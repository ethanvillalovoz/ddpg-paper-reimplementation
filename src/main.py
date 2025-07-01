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
from tensorflow.initializers import random_uniform

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
    """
    Replay buffer for storing and sampling experience tuples.
    Used in DDPG to enable off-policy learning by sampling random batches of past experiences.
    """
    def __init__(self, max_size, input_shape, n_actions):
        # Maximum number of transitions to store in the buffer
        self.memory_size = max_size
        # Counter to keep track of the number of transitions added
        self.memory_counter = 0
        # Memory arrays for states, new states, actions, rewards, and terminal flags
        self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)  # Fixed dtype typo

    def store_transition(self, state, action, reward, new_state, done):
        """
        Store a new experience in the buffer, overwriting the oldest if full.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            new_state (np.ndarray): Next state after action.
            done (bool): Whether the episode ended after this transition.
        """
        index = self.memory_counter % self.memory_size  # Circular buffer index
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = 1 - int(done)  # 0 if done, 1 otherwise
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            Tuple of (states, actions, rewards, new_states, terminals)
        """
        max_mem = min(self.memory_counter, self.memory_size)  # Only sample from filled part
        batch = np.random.choice(max_mem, batch_size)  # Random indices

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
    
class Actor(object):
    """
    Actor network for DDPG.
    Responsible for learning the deterministic policy (mapping states to actions).
    """
    def __init__(self, lr, n_actions, name, input_dims, sess, fcl_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        # Learning rate for optimizer
        self.lr = lr
        # Number of actions in the environment
        self.n_actions = n_actions
        # Name scope for the network (useful for TensorFlow variable scoping)
        self.name = name
        # Input dimensions (state space)
        self.input_dims = input_dims
        # TensorFlow session
        self.sess = sess
        # Number of units in the first fully connected layer
        self.fcl_dims = fcl_dims
        # Number of units in the second fully connected layer
        self.fc2_dims = fc2_dims
        # Action bound for scaling output actions
        self.action_bound = action_bound
        # Batch size for training
        self.batch_size = batch_size
        # Directory to save checkpoints
        self.chkpt_dir = chkpt_dir

        # Build the actor network
        self.build_network()
        # Get all trainable variables in this scope
        self.params = tf.trainable_variables(scope=self.name)
        # TensorFlow saver for saving/loading checkpoints
        self.saver = tf.train.Saver()
        # Path to save checkpoints
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name + '_ddpg.ckpt')

        # Compute unnormalized gradients of the actor loss w.r.t. actor parameters
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        # Normalize gradients by batch size
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        # Optimizer operation for applying gradients
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        """
        Build the TensorFlow computation graph for the actor network.
        """
        with tf.variable_scope(self.name):
            # Placeholder for input states
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            # Placeholder for action gradients (from the critic)
            self.action_gradient = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='action_gradients')

            # First dense layer with batch normalization and ReLU activation
            f1 = 1 / np.sqrt(self.fcl_dims)
            dense1 = tf.layers.dense(self.input, units=self.fcl_dims, kernel_initializer=random_uniform(-f1, f1), bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            # Second dense layer with batch normalization and ReLU activation
            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims, kernel_initializer=random_uniform(-f2, f2), bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            # Output layer with tanh activation, scaled by action_bound
            f3 = 0.003
            mu = tf.layers.dense(layer2_activation, units=self.n_actions, activation=tf.nn.tanh, kernel_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3))

            # Scale output to action bounds
            self.mu = tf.multiply(mu, self.action_bound, name='scaled_mu')

    def predict(self, inputs):
        """
        Predict actions given input states.
        Args:
            inputs (np.ndarray): Batch of input states.
        Returns:
            np.ndarray: Predicted actions.
        """
        return self.sess.run(self.mu, feed_dict={self.input: inputs})
    
    def train(self, inputs, gradients):
        """
        Train the actor network using the provided action gradients.
        Args:
            inputs (np.ndarray): Batch of input states.
            gradients (np.ndarray): Gradients from the critic.
        """
        self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action_gradient: gradients})

    def save_checkpoint(self):
        """
        Save the actor network parameters to disk.
        """
        print('... saving checkpoint for actor ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the actor network parameters from disk.
        """
        print('... loading checkpoint for actor ...')
        self.saver.restore(self.sess, self.checkpoint_file)
        print('... loaded checkpoint for actor ...')

class Critic(object):
    """
    Critic network for DDPG.
    Responsible for estimating the Q-value (expected return) for state-action pairs.
    """
    def __init__(self, lr, n_actions, name, input_dims, sess, fcl_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        # Learning rate for optimizer
        self.lr = lr
        # Number of actions in the environment
        self.n_actions = n_actions
        # Name scope for the network (useful for TensorFlow variable scoping)
        self.name = name
        # Input dimensions (state space)
        self.input_dims = input_dims
        # TensorFlow session
        self.sess = sess
        # Number of units in the first fully connected layer
        self.fcl_dims = fcl_dims
        # Number of units in the second fully connected layer
        self.fc2_dims = fc2_dims
        # Action bound for scaling output actions (not directly used in critic)
        self.action_bound = action_bound
        # Batch size for training
        self.batch_size = batch_size
        # Directory to save checkpoints
        self.chkpt_dir = chkpt_dir

        # Build the critic network
        self.build_network()
        # Get all trainable variables in this scope
        self.params = tf.trainable_variables(scope=self.name)
        # TensorFlow saver for saving/loading checkpoints
        self.saver = tf.train.Saver()
        # Path to save checkpoints
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name + '_ddpg.ckpt')

        # Optimizer operation for minimizing the critic loss
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Compute gradients of Q-values with respect to actions (for actor update)
        self.action_gradients = tf.gradients(self.q, self.action, name='action_gradients')

    def build_network(self):
        """
        Build the TensorFlow computation graph for the critic network.
        """
        with tf.variable_scope(self.name):
            # Placeholder for input states
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            # Placeholder for input actions
            self.action = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='actions')
            # Placeholder for target Q values
            self.q_target = tf.placeholder(tf.float32, shape=[None, 1], name='target')

            # First dense layer with batch normalization and ReLU activation
            f1 = 1 / np.sqrt(self.fcl_dims)
            dense1 = tf.layers.dense(self.input, units=self.fcl_dims, kernel_initializer=random_uniform(-f1, f1), bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            # Second dense layer with batch normalization and ReLU activation
            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims, kernel_initializer=random_uniform(-f2, f2), bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)

            # Process action input through a dense layer
            action_in = tf.layers.dense(self.action, units=self.fc2_dims, activation='relu')

            # Combine state and action pathways
            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)

            # Output Q-value layer with L2 regularization
            f3 = 0.003
            self.q = tf.layers.dense(
                state_actions,
                units=1,
                kernel_initializer=random_uniform(-f3, f3),
                bias_initializer=random_uniform(-f3, f3),
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                name='q_value'
            )
            # Mean squared error loss for critic
            self.loss = tf._losses.mean_squared_error(self.target, self.q, name='critic_loss')

    def predict(self, inputs, actions):
        """
        Predict Q values given input states and actions.
        Args:
            inputs (np.ndarray): Batch of input states.
            actions (np.ndarray): Batch of actions.
        Returns:
            np.ndarray: Predicted Q values.
        """
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.action: actions})
    
    def train(self, inputs, actions, q_target):
        """
        Train the critic network using the provided states, actions, and target Q values.
        Args:
            inputs (np.ndarray): Batch of input states.
            actions (np.ndarray): Batch of actions.
            q_target (np.ndarray): Target Q values.
        """
        self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action: actions, self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        """
        Get the gradients of the Q values with respect to the actions.
        Args:
            inputs (np.ndarray): Batch of input states.
            actions (np.ndarray): Batch of actions.
        Returns:
            np.ndarray: Action gradients.
        """
        return self.sess.run(self.action_gradients, feed_dict={self.input: inputs, self.action: actions})
    
    def save_checkpoint(self):
        """
        Save the critic network parameters to disk.
        """
        print('... saving checkpoint for critic ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the critic network parameters from disk.
        """
        print('... loading checkpoint for critic ...')
        self.saver.restore(self.sess, self.checkpoint_file)
        print('... loaded checkpoint for critic ...')
