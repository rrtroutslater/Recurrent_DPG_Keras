from ACBase import *


class CriticRDPG(ACBase):
    def __init__(self,
                 session,
                 obs_dim=32,
                 act_dim=3,
                 learning_rate=0.005,
                 training=True,
                 test_mode=False,
                 ):

        tf.compat.v1.disable_eager_execution()
        self.set_network_type("critic")

        self.sess = session
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.training = training
        self.learning_rate = learning_rate
        self.test_mode = test_mode

        self.lstm_units = 16

        # placeholders for tracking hidden state over variable-length episodes
        self.h_ph = tf.keras.backend.placeholder(
            shape=[1, self.lstm_units])
        self.c_ph = tf.keras.backend.placeholder(
            shape=[1, self.lstm_units])
        self.h_ph_t = tf.keras.backend.placeholder(
            shape=[None, 1, self.lstm_units], dtype=tf.float32)
        self.c_ph_t = tf.keras.backend.placeholder(
            shape=[None, 1, self.lstm_units], dtype=tf.float32)

        # hidden and carry state numpy arrays
        self.h_prev = None
        self.c_prev = None
        self.h_prev_t = None
        self.c_prev_t = None

        # critic, Q(o, mu(o))
        self.net, self.act_in, self.obs_in, self.q, self.critic_h_sequence, self.critic_h, self.critic_c = self.make_Q_net()
        self.net_weights = self.net.trainable_weights
        self.net.compile(loss='mean_squared_error', optimizer='adam')

        # loss, used for grad calcs
        self.y_ph = tf.keras.backend.placeholder(shape=[None, None, 1])
        self.loss = tf.reduce_sum(tf.reduce_sum(
            tf.square(self.q - self.y_ph), axis=0), axis=0)

        # critic target, Q'(o', mu(o'))
        self.net_t, self.act_in_t, self.obs_in_t,  self.q_t, self.critic_h_t_sequence, self.critic_h_t, self.critic_c_t = self.make_Q_net()
        self.net_t_weights = self.net_t.trainable_weights

        # gradients
        self.dL_dWq, self.dQ_da, self.dL_do = self.initialize_gradients()
        self.grads = {
            "dL/dWq = dL/dq * dQ/dWq": self.dL_dWq,
            "dQ/da": self.dQ_da,
            "dL/do = dL/dq * dQ/do": self.dL_do,
        }

        self.sess.run(tf.compat.v1.global_variables_initializer())
        return

    def sample_q(self,
                 obs,
                 act,
                 ):
        """
        sample Q(o, a) value

        inputs: 
            obs: observations, numpy array of shape (N, num_timestep, obs_dim)
            act: actions, numpy array of shape (N, num_timestep, act_dim)

        returns:
            q, numpy array of shape (N, num_timestep, 1)
        """
        if self.test_mode:
            print('\nhidden state before forward pass:\n', self.h_prev)

        q, h_prev, c_prev = self.sess.run(
            [self.q, self.critic_h, self.critic_c],
            feed_dict={
                self.obs_in: obs,
                self.act_in: act,
            }
        )

        if self.test_mode:
            # get entire hidden state history
            hidden_state_history = self.sess.run(
                [self.critic_h_sequence],
                feed_dict={
                    self.obs_in: obs,
                    self.act_in: act,
                }
            )

        # store the final hidden states from time t to be used as inital hidden states at time t+1
        self.h_prev = h_prev
        self.c_prev = c_prev

        if self.test_mode:
            print('\nq:\n', q)
            print('\ncarry state:\n', self.c_prev)
            print('\nhidden state after forward pass:\n', self.h_prev)
            print('\nhidden state history:\n', hidden_state_history)
        return q

    def sample_q_target(self,
                        obs,
                        act,
                        ):
        """
        sample Q'(o', a') value from TARGET network

        inputs: 
            obs: observations, numpy array of shape (N, num_timestep, obs_dim)
            act: actions, numpy array of shape (N, num_timestep, act_dim)

        returns:
            q, numpy array of shape (N, num_timestep, 1)
        """
        if self.test_mode:
            print('\nhidden state before forward pass:\n', self.h_prev_t)

        q_t, h_prev_t, c_prev_t = self.sess.run(
            [self.q_t, self.critic_h_t, self.critic_c_t],
            feed_dict={
                self.obs_in_t: obs,
                self.act_in_t: act,
            }
        )

        if self.test_mode:
            # get entire hidden state history
            hidden_state_history = self.sess.run(
                [self.critic_h_t_sequence],
                feed_dict={
                    self.obs_in_t: obs,
                    self.act_in_t: act,
                }
            )

        # store the final hidden states from time t to be used as inital hidden states at time t+1
        self.h_prev_t = h_prev_t
        self.c_prev_t = c_prev_t

        if self.test_mode:
            print('\nq:\n', q_t)
            print('\ncarry state:\n', self.c_prev_t)
            print('\nhidden state after forward pass:\n', self.h_prev_t)
            print('\nhidden state history:\n', hidden_state_history)
        return q_t

    def get_dQ_da_critic(self,
                         obs,
                         act,
                         ):
        """
        compute gradient of objective w.r.t. action, a = mu(o)

        inputs:
            obs: extracted features, numpy array,
                shape (N, num_timestep, obs_dim)
            act: actions, numpy array,
                shape (N, num_timestep, act_dim)

        returns: 
            dJ/da, gradient of objective w.r.t. action
        """
        dQ_da = self.sess.run(
            self.dQ_da,
            feed_dict={
                self.obs_in: obs,
                self.act_in: act,
            }
        )
        return dQ_da[0]

    def get_dL_do_critic(self,
                         labels,
                         obs,
                         act,
                         ):
        """
        compute gradient of loss function w.r.t. observation, to be backpropagated through encoder

        inputs:
            labels: training labels for DPG Bellman error, numpy array,
                shape (N, num_timestep, 1)
            obs: extracted features, numpy array,
                shape (N, num_timestep, obs_dim)
            act: actions, numpy array,
                shape (N, num_timestep, act_dim)

        returns: 
            dL/do, gradient of Bellman Loss w.r.t. observation
        """
        dL_do = self.sess.run(
            self.dL_do,
            feed_dict={
                self.y_ph: labels,
                self.obs_in: obs,
                self.act_in: act,
            }
        )
        return dL_do[0]

    def make_Q_net(self,):
        """
        """
        act_in = tf.keras.layers.Input(
            shape=[None, self.act_dim]
        )

        obs_in = tf.keras.layers.Input(
            shape=[None, self.obs_dim]
        )

        act_obs_in = tf.concat([act_in, obs_in], axis=2)

        lstm_sequence, h, c = keras.layers.LSTM(
            units=self.lstm_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            return_state=True,
            stateful=False,
        )(act_obs_in)

        q = keras.layers.Dense(
            units=1,
            activation='sigmoid'
        )(lstm_sequence)

        model = keras.Model(inputs=[act_in, obs_in], outputs=q)
        return model, act_in, obs_in, q, lstm_sequence, h, c

    def initialize_gradients(self,):
        """
        initialize gradients of Q(o, a) w.r.t. obs and action

        returns:
            placeholder for upstream gradient of Bellman Error w.r.t. Q function output
            gradient of Bellman Error w.r.t. Q network weights
            gradient of Bellman Error w.r.t. action input 
                backpropagated through actor and feature extractor during training
            gradient of Bellman Error w.r.t. observation input
                backpropagated through feature extractor during training
        """
        # list of gradients of Bellman Error w.r.t. Q function weights: dL/dWq = dL/dq * dQ/dWq
        dL_dWq = tf.gradients(
            self.loss,
            self.net_weights,
        )

        # gradient Q function w.r.t. Q function action input: dQ/da
        dQ_da = tf.gradients(
            self.q,
            self.act_in,
        )

        # gradient of Bellman Error w.r.t. Q function obs input: dL/do = dL/dq * dq/do
        dL_do = tf.gradients(
            self.loss,
            self.obs_in
        )

        return dL_dWq, dQ_da, dL_do


if __name__ == "__main__":

    session = tf.compat.v1.Session()

    critic = CriticRDPG(
        session,
        training=True,
        test_mode=True,
    )

    np.random.seed(0)

    # test forward pass on series of single observations
    if 0:
        for i in range(0, 4):
            print('\n--------------')
            obs = np.random.randn(1, 1, 32)
            act = np.random.randn(1, 1, 3)
            q = critic.sample_q(obs, act)

    # make a stack of unique observations:
    obs = np.random.randn(1, 4, 32)
    act = np.random.randn(1, 4, 3)

    # test forward pass on batch with hidden state propagation
    if 1:
        # target
        print("\ncritic target:\n")
        q = critic.sample_q_target(obs, act)
        q_1 = critic.sample_q_target(np.random.randn(
            1, 1, 32), np.random.randn(1, 1, 3))

        print("\ncritic:\n")
        # actor
        q = critic.sample_q(obs, act)
        q_1 = critic.sample_q(np.random.randn(
            1, 1, 32), np.random.randn(1, 1, 3))

    # test gradients
    if 1:
        # test dQ_da
        act = np.random.randn(1, 4, 3)
        dQ_da = critic.get_dQ_da_critic(obs, act)
        print('\ndL/da:\n', dQ_da)

        # test dL_do
        labels = np.random.randn(1, 4, 1)
        dL_do = critic.get_dL_do_critic(labels, obs, act)
        print('\ndL/do:\n', dL_do)

    if 0:
        # test q
        act = np.random.randn(1, 4, 3)
        q = critic.sample_q(obs, act)
        print('\nq:\n', q)
        q_t = critic.sample_q_target(obs, act)
        print('\nq\':\n', q_t)

    # test training
    if 0:
        act = np.random.randn(1, 4, 3)
        obs = np.random.randn(1, 4, 32)
        y = np.random.randn(1, 4, 1)
        loss = critic.net.train_on_batch([act, obs], y)
        print(loss)

    pass
