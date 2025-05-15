class DDPGAgent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000, batch_size=64, layer1_size=400, layer2_size=300):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.max_size = max_size
        self.batch_size = batch_size
        self.env = env
        self.sess = tf.Session()
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.actor = ActorNetwork(alpha, n_actions, input_dims, 'Actor', self.sess, layer1_size, layer2_size, action_bound=env.action_space.high)
        self.critic = Critic(beta, n_actions, input_dims, 'Critic', self.sess, layer1_size, layer2_size, action_bound=env.action_space.high)
        self.target_actor = ActorNetwork(alpha, n_actions, input_dims, 'TargetActor', self.sess, layer1_size, layer2_size, action_bound=env.action_space.high)
        self.target_critic = Critic(beta, n_actions, input_dims, 'TargetCritic', self.sess, layer1_size, layer2_size, action_bound=env.action_space.high)
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions))

        self.target_critic_params = tf.trainable_variables(scope='TargetCritic')
        self.update_critic = [
            self.target_critic_params[i].assign(
                tf.multiply(self.critic.params[i], self.tau) +
                tf.multiply(self.target_critic.params[i], 1 - self.tau)
            ) for i in range(len(self.target_critic_params))
        ]

        self.update_actor = [
            self.target_actor.params[i].assign(
                tf.multiply(self.actor.params[i], self.tau) +
                tf.multiply(self.target_actor.params[i], 1 - self.tau)
            ) for i in range(len(self.target_actor.params))
        ]
        self.sess.run(tf.global_variables_initializer())
        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, state_, done):
        # Extract observation if input is a tuple (obs, info)
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state_, tuple):
            state_ = state_[0]
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        state_ = np.array(state_, dtype=np.float32)
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        # Handle Gym's new API: observation may be a tuple (obs, info)
        if isinstance(observation, tuple):
            obs = observation[0]
        else:
            obs = observation
        state = np.array(obs)[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise
        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))
        target = []
        for i in range(self.batch_size):
            target.append(reward[i] + self.gamma * critic_value_[i] * done[i])
        target = np.reshape(target, (self.batch_size, 1))
        _ = self.critic.train(state, action, target)
        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
