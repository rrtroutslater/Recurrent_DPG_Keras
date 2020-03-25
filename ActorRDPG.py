from ACBase import *


class ActorRDPG(ACBase):
    def __init__(self,
                 session,
                 obs_dim=32,
                 act_dim=3,
                 learning_rate=0.005,
                 training=True,
                 test_mode=False,
                 ):

        tf.compat.v1.disable_eager_execution()
        self.set_network_type("actor")

        self.sess = session
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.learning_rate = learning_rate
        self.training = training
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
        # self.h_prev = np.random.randn(1, self.lstm_units)
        # self.c_prev = np.random.randn(1, self.lstm_units)
        # self.h_prev_t = np.random.randn(None, 1, self.lstm_units)
        # self.c_prev_t = np.random.randn(None, 1, self.lstm_units)
        self.h_prev = None
        self.c_prev = None
        self.h_prev_t = None
        self.c_prev_t = None

        # actor, mu(o)
        self.net, self.obs_in, self.act, self.actor_h_sequence, self.actor_h, self.actor_c = self.make_act_net()
        self.net_weights = self.net.trainable_weights
        print('obs in shape:\t', self.obs_in.get_shape())
        print('act shape:\t', self.obs_in.get_shape())
        print('h shape:\t', self.actor_h.get_shape())
        print('h seq shape:\t', self.actor_h_sequence.get_shape())
        print('c shape:\t', self.actor_c.get_shape())

        # actor target, mu'(o)
        self.net_t, self.obs_t_in, self.act_t, self.actor_h_t_sequence, self.actor_h_t, self.actor_c_t = self.make_act_net()
        self.net_t_weights = self.net_t.trainable_weights

        # gradients
        self.dQ_da, self.dJ_dWa, self.dJ_do = self.initialize_gradients()
        self.grads = {
            "dQ/da": self.dQ_da,
            "dJ/dWa = 1/NT (dQ/da * dmu/dWa)": self.dJ_dWa,
            "dJ/do = dQ/da * dmu/do": self.dJ_do,
        }

        # gradient step
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.grad_step = self.optimizer.apply_gradients(
            zip(self.dJ_dWa, self.net_weights)
        )

        self.sess.run(tf.compat.v1.global_variables_initializer())
        return

    def make_act_net(self):
        """
        define actor network architecture

        returns:
            keras model 
            input layer
            action output
            hidden state h
            hidden state c
        """
        obs_in = keras.layers.Input(
            shape=[None, self.obs_dim],
        )

        lstm_out, h, c = keras.layers.LSTM(
            units=self.lstm_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            return_state=True,
            stateful=False,
            # )(obs_in, initial_state=[self.h_ph, self.c_ph])
        )(obs_in)

        act = keras.layers.Dense(
            units=self.act_dim,
            activation="tanh",
        )(lstm_out)

        # act = keras.layers.Dense(
        #     units=self.act_dim,
        #     activation="tanh",
        # )(h)

        model = keras.Model(inputs=obs_in, outputs=act)
        return model, obs_in, act, lstm_out, h, c

    def sample_act(self,
                   obs,
                   add_noise=False
                   ):
        """
        sampe action, a = mu(obs)

        input:
            obs: observation, numpy array of shape (N, num_timestep, obs_dim)
            add_noise: flag indicating whether to add noise to action, used for exploration
                of state/action space during training

        returns:
            action, numpy array of shape (N, num_timestep, act_dim)
        """
        if self.test_mode:
            print('\nhidden state before forward pass:\n', self.h_prev)

        if self.h_prev is None:
            self.h_prev = np.random.randn(obs.shape[0], 1, self.lstm_units)
        if self.c_prev is None:
            self.c_prev = np.random.randn(obs.shape[0], 1, self.lstm_units)

        act, h_prev, c_prev = self.sess.run(
            [self.act, self.actor_h, self.actor_c],
            feed_dict={
                self.obs_in: obs,
                # self.h_ph: self.h_prev,
                # self.c_ph: self.c_prev,
            }
        )

        if self.test_mode:
            # get entire hidden state history
            hidden_state_history = []
            hidden_state_history = self.sess.run(
                [self.actor_h_sequence],
                feed_dict={
                    self.obs_in: obs,
                    # self.h_ph: self.h_prev,
                    # self.c_ph: self.c_prev,
                }
            )

        # store the final hidden states from time t to be used as inital hidden states at time t+1
        self.h_prev = h_prev
        self.c_prev = c_prev

        if self.test_mode:
            print('\nact:\n', act)
            print('\ncarry state:\n', self.c_prev)
            print('\nhidden state after forward pass:\n', self.h_prev)
            print('\nhidden state history:\n', hidden_state_history)

        return act

    def sample_act_target(self,
                          obs
                          ):
        """
        sample action from target network, a' = mu'(obs')

        input:
            obs: observation, numpy array of shape (N, num_timestep, obs_dim)

        returns:
            action, numpy array of shape (N, num_timestep, act_dim)
        """
        if self.test_mode:
            print('\nhidden state before forward pass:\n', self.h_prev_t)

        if self.h_prev_t is None:
            self.h_prev_t = np.random.randn(obs.shape[0], 1, self.lstm_units)
        if self.c_prev is None:
            self.c_prev_t = np.random.randn(obs.shape[0], 1, self.lstm_units)

        act_t, h_prev_t, c_prev_t = self.sess.run(
            [self.act_t, self.actor_h_t, self.actor_c_t],
            feed_dict={
                self.obs_t_in: obs,
                # self.h_ph: self.h_prev_t,
                # self.c_ph: self.c_prev_t,
            }
        )

        hidden_state_history = []
        # get entire hidden state history, for testing
        if self.test_mode:
            hidden_state_history = self.sess.run(
                [self.actor_h_t_sequence],
                feed_dict={
                    self.obs_t_in: obs,
                    # self.h_ph: self.h_prev_t,
                    # self.c_ph: self.c_prev_t,
                }
            )

        # store the final hidden states from time t to be used as inital hidden states at time t+1
        self.h_prev_t = h_prev_t
        self.c_prev_t = c_prev_t

        if self.test_mode:
            print('\nact:\n', act_t)
            print('\ncarry state:\n', self.c_prev_t)
            print('\nhidden state after forward pass:\n', self.h_prev_t)
            print('\nhidden state history:\n', hidden_state_history)
        return act_t

    def initialize_gradients(self):
        """
        initialize necessary gradients
        tf.gradients(
            y,  
            x,
            upstream_grads, (eg, dz/dy)
        )
        computes dy/dx * upstream_grads

        returns:
            placeholder for upstream gradient of RL objective w.r.t. action: dJ/da
            gradient of objective w.r.t. network weights, a list
            gradient of objective w.r.t. observation input
        """
        # placeholder for result of backpropagation of objective gradient through Critic Q(s,a)
        dQ_da = tf.keras.backend.placeholder(
            shape=[None, None, self.act_dim],
            dtype=tf.float32,
        )
        print("dQ_da shape at declaration:\t", dQ_da.get_shape())

        # list of gradients of objective w.r.t. actor (mu) weights: dJ/dWa = dmu/dWa * dQ/da, mu(f) = a.
        # comes from objective maximization via policy gradient.
        dJ_dWa = tf.gradients(
            self.act,
            self.net_weights,
            -dQ_da,
        )

        # gradient of objective w.r.t. obs input: dJ/do = dmu/do * dJ/da, mu(f) = a
        dJ_do = tf.gradients(
            self.act,
            self.obs_in,
            -dQ_da,
        )
        # obs extractor is not recurrent, so sum up gradients from all timesteps
        # dJ_do = tf.reduce_sum(dJ_do, axis=2)

        return dQ_da, dJ_dWa, dJ_do

    def apply_gradients(self,
                        obs,
                        dQ_da,
                        num_step=1,
                        ):
        """
        compute and apply gradient of objective w.r.t. network weights

        inputs:
            obs: series of observations, numpy array, 
                shape (N, num_timestep, obs_dim)
            dQ_da: gradient of loss with respect to actor output, numpy array, 
                shape (N, num_timestep, act_dim)
            num_step: number of gradient steps to perform
        """
        print("-----------------------------")
        print("h_ph shape:\t", self.h_ph.get_shape())
        print("c_ph shape:\t", self.c_ph.get_shape())
        print("h shape:\t", self.actor_h.get_shape())
        print("c shape:\t", self.actor_c.get_shape())
        print("dQ_da expected:\t", self.dQ_da.get_shape())
        print("dQ_da input:\t", dQ_da.shape)
        print("obs expected:\t", self.obs_in.get_shape())
        print("obs input:\t", obs.shape)

        if self.h_prev is None:
            self.h_prev = np.random.randn(obs.shape[0], 1, self.lstm_units)
        if self.c_prev is None:
            self.c_prev = np.random.randn(obs.shape[0], 1, self.lstm_units)
        print("h_prev shape:\t", self.h_prev.shape)
        print("c_prev shape:\t", self.c_prev.shape)

        for i in range(0, num_step):
            if self.test_mode:
                print('----------\nweights before update:',
                      self.net_weights[0].eval(session=self.sess))

            # act forward propagation
            act = self.sess.run(
                self.act,
                feed_dict={
                    self.obs_in: obs,
                    # self.h_ph: self.h_prev,
                    # self.c_ph: self.c_prev,
                }
            )
            print("act:\n", act)
            print("act shape:\t", act.shape)

            # grad calculation
            dJ_dWa = self.sess.run(
                self.dJ_dWa,
                feed_dict={
                    self.obs_in: obs,
                    self.dQ_da: dQ_da,
                    # self.h_ph: self.h_prev,
                    # self.c_ph: self.c_prev,
                }
            )
            print(len(dJ_dWa))
            print(len(dJ_dWa[0]))
            print(dJ_dWa[0])
            # for i in range(len(dJ_dWa)):
            #     print(dJ_dWa.shape)

            # application of gradient
            _ = self.sess.run(
                [self.grad_step],
                feed_dict={
                    self.obs_in: obs,
                    self.dQ_da: dQ_da,
                    # self.h_ph: self.h_prev,
                    # self.c_ph: self.c_prev,
                }
            )

            if self.test_mode:
                print('\nweights after update:',
                      self.net_weights[0].eval(session=self.sess))
                print('obs shape:\t', obs.shape)
                print('dQ_da shape:\t', dQ_da.shape)
                for i in range(0, len(dJ_dWa)):
                    print(dJ_dWa[i].shape)
        return

    def get_dJ_do_actor(self,
                        obs,
                        dQ_da,
                        ):
        """
        compute gradient of objective w.r.t. observation. used for backpropagation through feature extractor.

        inputs:
            obs: observation, numpy array, 
                shape (N, num_timestep, obs_dim)
            dQ_da: gradient of objective w.r.t. action a = mu(o), numpy array, 
                shape (N, num_timestep, act_dim)

        returns:
            dJ/do, gradient of objective w.r.t. observation input
        """

        if self.h_prev is None:
            self.h_prev = np.random.randn(obs.shape[0], 1, self.lstm_units)
        if self.c_prev is None:
            self.c_prev = np.random.randn(obs.shape[0], 1, self.lstm_units)

        dJ_do = self.sess.run(
            self.dJ_do,
            feed_dict={
                self.dQ_da: dQ_da,
                self.obs_in: obs,
                # self.h_ph: self.h_prev,
                # self.c_ph: self.c_prev,
            }
        )
        return dJ_do[0]

    def propagate_actor_episode(self,
                                obs,
                                obs_target,
                                ):
        """
        forward propagate data before an episode in order to update hidden states for network and target
        """
        _ = self.sample_act(obs)
        _ = self.sample_act_target(obs_target)
        return


if __name__ == "__main__":

    session = tf.compat.v1.Session()

    # testing forward pass with hidden state maintenance for a sequence of observations (ie, inference time)
    if 0:
        actor = ActorRDPG(
            session,
            training=False,
            test_mode=True,
        )
        np.random.seed(0)
        o = np.random.randn(1, 1, 32)
        for i in range(0, 4):
            print('----------------------------')
            print('\nobs:\n', o[:10])
            actor.sample_act_target(o)

    # testing forward pass with hidden state maintenance for a batch of observations (ie, training time)
    if 0:
        lstm_horizon = 5
        actor = ActorRDPG(
            session,
            training=True,
            test_mode=True,
        )
        np.random.seed(0)
        o = np.random.randn(1, lstm_horizon, 32)
        for i in range(0, 2):
            print('----------------------------')
            # print('\nobs:\n', o[:10])
            actor.sample_act(o)

    # test gradient update
    if 1:
        N = 3
        lstm_horizon = 4
        actor = ActorRDPG(
            session,
            training=True,
            test_mode=True,
        )
        np.random.seed(0)
        o = np.random.randn(3, lstm_horizon, 32)
        dQ_da = np.random.randn(3, lstm_horizon, 3)
        actor.apply_gradients(o, dQ_da, num_step=2)

    # test gradient
    if 0:
        lstm_horizon = 4
        N = 3
        actor = ActorRDPG(
            session,
            training=True,
            test_mode=True,
        )
        o = np.random.randn(N, lstm_horizon, 32)
        dQ_da = np.random.randn(N, lstm_horizon, 3)
        dJ_do = actor.get_dJ_do_actor(o, dQ_da)
        print('\ndJ/doa:\n', dJ_do)
        print('\ndJ/doa shape:\n', dJ_do.shape)

        # 1005197283
    pass


"""
backpropagation through time
    - requires summation of:
        dQ/da * dmu/dW
    over an episode (so at each timestep, Q(o, mu(o)) and mu(o) have been calculated).
    When computing this gradient, if dQ/da has shape [1, T, 32], grads are computed just fine,
    although they are returned as tensors the shape as the matrices in the architecture, which 
    may or may not be correct. Should they be (N, T, dim) or just (dim)? (meaning the summation
    has already taken place)?

    When computing this gradient, if dQ/da has shape [N, T, 32], nothing works

"""
