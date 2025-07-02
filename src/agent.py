import numpy as np
import tensorflow as tf
from ou_noise import OUActionNoise
from replay_buffer import ReplayBuffer
from networks import Actor, Critic
from typing import Any, Optional, Tuple


class Agent:
    """
    DDPG Agent using TensorFlow 2.x and Keras.
    Manages the actor/critic networks, target networks, replay buffer,
    and learning process.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        input_dims: list,
        tau: float,
        env: Any,
        gamma: float = 0.99,
        n_actions: int = 1,
        max_size: int = 1000000,
        layer1_size: int = 400,
        layer2_size: int = 300,
        batch_size: int = 64,
        actor_path: str = "actor.h5",
        critic_path: str = "critic.h5",
        target_actor_path: str = "target_actor.h5",
        target_critic_path: str = "target_critic.h5",
    ):
        # Discount factor for future rewards
        self.gamma = gamma
        # Soft update parameter for target networks
        self.tau = tau
        # Experience replay buffer
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        # Batch size for learning
        self.batch_size = batch_size

        # Actor and Critic networks
        self.actor = Actor(
            n_actions, layer1_size, layer2_size, env.action_space.high[0]
        )
        self.critic = Critic(layer1_size, layer2_size, n_actions)
        # Target networks for stability
        self.target_actor = Actor(
            n_actions, layer1_size, layer2_size, env.action_space.high[0]
        )
        self.target_critic = Critic(layer1_size, layer2_size, n_actions)

        # Compile models with Adam optimizers
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=alpha)
        )
        self.target_critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=beta)
        )

        # Ornstein-Uhlenbeck noise for exploration
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        # Hard update target networks to match main networks at start
        self.update_network_parameters(tau=1.0)

        # Model save/load paths
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.target_actor_path = target_actor_path
        self.target_critic_path = target_critic_path

    def update_network_parameters(self, tau: Optional[float] = None) -> None:
        """
        Soft update target network parameters.
        Args:
            tau (float or None): Soft update coefficient. If None, uses self.tau.
        """
        if tau is None:
            tau = self.tau
        # Update target actor
        weights = []
        targets = []
        for a, b in zip(self.actor.weights, self.target_actor.weights):
            weights.append(a)
            targets.append(b)
        for i in range(len(weights)):
            targets[i].assign(tau * weights[i] + (1 - tau) * targets[i].numpy())
        # Update target critic
        weights = []
        targets = []
        for a, b in zip(self.critic.weights, self.target_critic.weights):
            weights.append(a)
            targets.append(b)
        for i in range(len(weights)):
            targets[i].assign(tau * weights[i] + (1 - tau) * targets[i].numpy())

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the replay buffer.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(
        self, observation: np.ndarray, evaluate: bool = False
    ) -> np.ndarray:
        """
        Choose an action based on the current state using the actor network,
        and add exploration noise for exploration.
        Args:
            observation (np.ndarray): Current state.
            evaluate (bool): If True, do not add noise (for evaluation).
        Returns:
            np.ndarray: Chosen action with (optional) exploration noise.
        """
        # Ensure observation is a 2D float32 array
        state = np.asarray(observation, dtype=np.float32)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        actions = self.actor(state)
        if not evaluate:
            actions += self.noise()
        return np.clip(actions[0], -self.actor.action_bound, self.actor.action_bound)

    def learn(self) -> Optional[Tuple[float, float]]:
        """
        Sample a batch from the replay buffer and update the actor and critic networks.
        Returns:
            tuple: (critic_loss, actor_loss) or None if not enough samples.
        """
        if self.memory.mem_cntr < self.batch_size:
            return None

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(
            self.batch_size
        )

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            critic_value_ = tf.squeeze(
                self.target_critic(new_states, target_actions), 1
            )
            target = rewards + self.gamma * critic_value_ * dones
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        # Actor update
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, new_policy_actions))
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        # Soft update target networks
        self.update_network_parameters()

        return float(critic_loss.numpy()), float(actor_loss.numpy())

    def save_models(self) -> None:
        """
        Save the actor and critic networks (and their targets) to disk.
        """
        self.actor.save_weights(self.actor_path)
        self.critic.save_weights(self.critic_path)
        self.target_actor.save_weights(self.target_actor_path)
        self.target_critic.save_weights(self.target_critic_path)

    def load_models(self) -> None:
        """
        Load the actor and critic networks (and their targets) from disk.
        """
        self.actor.load_weights(self.actor_path)
        self.critic.load_weights(self.critic_path)
        self.target_actor.load_weights(self.target_actor_path)
        self.target_critic.load_weights(self.target_critic_path)
