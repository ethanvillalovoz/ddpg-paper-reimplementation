class ActorNetwork(object):
    def __init__(self, lr, n_actions, input_dims, name, sess, fcl_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
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
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
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

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.actor_gradients = tf.gradients(self.q, self.action, name='actor_gradients')

    def build_network(self):
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
