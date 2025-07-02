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
    """Ornstein-Uhlenbeck process for exploration noise."""
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        return (self.state_memory[batch],
                self.action_memory[batch],
                self.reward_memory[batch],
                self.new_state_memory[batch],
                self.terminal_memory[batch])

class Actor(tf.keras.Model):
    """Actor network."""
    def __init__(self, n_actions, fc1_dims, fc2_dims, action_bound):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.out = tf.keras.layers.Dense(n_actions, activation='tanh')
        self.action_bound = action_bound

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.out(x)
        return x * self.action_bound

class Critic(tf.keras.Model):
    """Critic network."""
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(Critic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

        self.action_fc = tf.keras.layers.Dense(fc2_dims, activation='relu')

    def call(self, state, action):
        x = self.fc1(state)
        x = self.fc2(x)
        a = self.action_fc(action)
        x = tf.keras.layers.Add()([x, a])
        x = tf.nn.relu(x)
        q = self.q(x)
        return q

class Agent:
    """DDPG Agent using TensorFlow 2.x and Keras."""
    def __init__(
        self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=1,
        max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = Actor(n_actions, layer1_size, layer2_size, env.action_space.high[0])
        self.critic = Critic(layer1_size, layer2_size, n_actions)
        self.target_actor = Actor(n_actions, layer1_size, layer2_size, env.action_space.high[0])
        self.target_critic = Critic(layer1_size, layer2_size, n_actions)

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1.0)  # Hard update at start

    def update_network_parameters(self, tau=None):
        """Soft update target network parameters."""
        if tau is None:
            tau = self.tau
        weights = []
        targets = []
        for a, b in zip(self.actor.weights, self.target_actor.weights):
            weights.append(a)
            targets.append(b)
        for i in range(len(weights)):
            targets[i].assign(tau * weights[i] + (1 - tau) * targets[i].numpy())
        weights = []
        targets = []
        for a, b in zip(self.critic.weights, self.target_critic.weights):
            weights.append(a)
            targets.append(b)
        for i in range(len(weights)):
            targets[i].assign(tau * weights[i] + (1 - tau) * targets[i].numpy())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation, evaluate=False):
        # Ensure observation is a 2D float32 array
        state = np.asarray(observation, dtype=np.float32)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        actions = self.actor(state)
        if not evaluate:
            actions += self.noise()
        return np.clip(actions[0], -self.actor.action_bound, self.actor.action_bound)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            critic_value_ = tf.squeeze(self.target_critic(new_states, target_actions), 1)
            target = rewards + self.gamma * critic_value_ * dones
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, new_policy_actions))
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Soft update target networks
        self.update_network_parameters()

    def save_models(self):
        self.actor.save_weights('actor.h5')
        self.critic.save_weights('critic.h5')
        self.target_actor.save_weights('target_actor.h5')
        self.target_critic.save_weights('target_critic.h5')

    def load_models(self):
        self.actor.load_weights('actor.h5')
        self.critic.load_weights('critic.h5')
        self.target_actor.load_weights('target_actor.h5')
        self.target_critic.load_weights('target_critic.h5')
