"""
ddpg_agent.py

TensorFlow 2.x version: DDPGAgent class for training and interacting with the environment.
"""

import numpy as np
import tensorflow as tf
print("Logical devices:", tf.config.list_logical_devices())
from .networks import ActorNetwork, Critic
from .buffer import ReplayBuffer
from .noise import OrnsteinUhlenbeckActionNoise

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent using TensorFlow 2.x.
    Handles training, action selection, and model saving/loading.
    """
    def __init__(
        self, alpha: float, beta: float, input_dims: list, tau: float, env,
        gamma: float = 0.99, n_actions: int = 2, max_size: int = 1000000,
        batch_size: int = 64, layer1_size: int = 400, layer2_size: int = 300
    ):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.max_size = max_size
        self.batch_size = batch_size
        self.env = env

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.actor = ActorNetwork(n_actions, layer1_size, layer2_size, input_dims, env.action_space.high)
        self.critic = Critic(layer1_size, layer2_size, input_dims, n_actions)
        self.target_actor = ActorNetwork(n_actions, layer1_size, layer2_size, input_dims, env.action_space.high)
        self.target_critic = Critic(layer1_size, layer2_size, input_dims, n_actions)
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions))

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)

        # Initialize target networks
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        """
        Soft update target networks.
        """
        if tau is None:
            tau = self.tau
        # Update actor
        for target_param, param in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        # Update critic
        for target_param, param in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        state = np.array(observation, dtype=np.float32)[np.newaxis, :]
        mu = self.actor(state)
        mu = mu.numpy()[0]
        noise = self.noise()
        mu_prime = mu + noise
        return np.clip(mu_prime, self.env.action_space.low, self.env.action_space.high)

    @tf.function
    def train_step(self, state, action, reward, new_state, done):
        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_state)
            target_critic_value = tf.squeeze(self.target_critic(new_state, target_actions), 1)
            y = reward + self.gamma * target_critic_value * (1 - done)
            critic_value = tf.squeeze(self.critic(state, action), 1)
            critic_loss = tf.keras.losses.MSE(y, critic_value)
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(state)
            actor_loss = -tf.reduce_mean(self.critic(state, new_policy_actions))
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        self.train_step(state, action, reward, new_state, done)
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
