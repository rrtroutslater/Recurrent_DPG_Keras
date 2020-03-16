from ACBase import *
from Encoder import make_encoder_net


class CriticRDPG(ACBase):
    def __init__ (self,
            session,
            img_dim=[16,90,3],
            obs_dim=32,
            act_dim=3,
            lstm_horizon=4,
            learning_rate=0.005,
            training=True,
        ):
        
        tf.compat.v1.disable_eager_execution()
        self.set_network_type("critic")

        self.sess = session
        self.img_dim = img_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if training:
            self.lstm_horizon = lstm_horizon
        else:
            self.lstm_horizon = 1
        self.training = training
        self.learning_rate = learning_rate

        # encoder, takes stack of range images, returns extracted features
        self.feature_extractor_net, self.img_in, self.obs = make_encoder_net(
            img_dim,
            obs_dim
        )
        self.feature_extractor_weights = self.feature_extractor_net.trainable_weights

        # critic, Q(o, mu(o))
        self.net, self.act_in, self.obs_in, self.q, self.critic_h, self.critic_c = self.make_Q_net()
        self.net_weights = self.net.trainable_weights
        self.net.compile(loss='mean_squared_error', optimizer='sgd')

        # critic target, Q'(o', mu(o'))
        self.net_t, self.act_in_t, self.obs_in_t,  self.q_t, self.critic_h_t, self.critic_c_t = self.make_Q_net()
        self.net_t_weights = self.net_t.trainable_weights

        # gradients
        self.dL_dq, self.dL_dWq, self.dL_da, self.dL_do = self.initialize_gradients()
        self.grads = {
            "dL/dq": self.dL_dq,
            "dL/dWq = dL/dq * dQ/dWq": self.dL_dWq,
            "dL/da = dL/dq * dQ/da": self.dL_da,
            "dL/do = dL/dq * dQ/do": self.dL_do,
        }

        # gradient step
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.grad_step = self.optimizer.apply_gradients(zip(self.dL_dWq, self.net_weights))

        self.sess.run(tf.compat.v1.global_variables_initializer())
        return

    def sample_obs(self, 
            img_in
        ):
        """
        sample an observation = features extracted from stack of range images

        inputs:
            img_in: stack of range images, numpy array of shape (N, num_timestep, img_dim)

        returns:
            obs, numpy array of shape (N, num_timestep, obs_dim)
        """
        obs = self.feature_extractor_net.predict(img_in)[0]
        return obs

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
        q = self.net.predict([act, obs])
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
        q = self.net_t.predict([act, obs])
        return q

    def get_dL_da(self,
            dL_dq,
            obs,
            act,
        ):
        """
        compute gradient of Bellman Error w.r.t. action, a = mu(o)

        inputs:
            dL_dq: upstream gradient of Bellman Error w.r.t. q value, numpy array,
                shape (N, num_timestep, 1)
            obs: extracted features, numpy array,
                shape (N, num_timestep, obs_dim)
            act: actions, numpy array,
                shape (N, num_timestep, act_dim)

        returns: 
            dL/da, gradient of Bellman Loss w.r.t. action
        """
        dL_da = self.sess.run(
            self.dL_da, 
            feed_dict={
                self.dL_dq: dL_dq,
                self.obs_in: obs,
                self.act_in: act,
            }
        )
        return dL_da[0]

    def get_dL_do(self,
            dL_dq,
            obs,
            act,
        ):
        """
        compute gradient of Bellman Error w.r.t. observation

        inputs:
            dL_dq: upstream gradient of Bellman Error w.r.t. q value, numpy array,
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
                self.dL_dq: dL_dq,
                self.obs_in: obs,
                self.act_in: act,
            }
        )
        return dL_do

    def make_Q_net(self,):
        """
        """

        act_in = tf.keras.layers.Input(
            shape=[self.lstm_horizon, self.act_dim]
        )

        obs_in = tf.keras.layers.Input(
            shape=[self.lstm_horizon, self.obs_dim]
        )

        act_obs_in = tf.concat([act_in, obs_in], axis=2)

        lstm_out, _, c = keras.layers.LSTM(
            units=32,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            return_state=True, 
            stateful=False,
        )(act_obs_in)

        q = keras.layers.Dense(
            units=1,
            activation='sigmoid'
        )(lstm_out)

        model = keras.Model(inputs=[act_in, obs_in], outputs=q)
        return model, act_in, obs_in, q, lstm_out, c

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

        # placeholder for gradient of Bellman Error w.r.t. Q function output
        dL_dq = tf.keras.backend.placeholder(
            shape=[None, self.lstm_horizon, 1],
            dtype=tf.float32,
        )

        # list of gradients of Bellman Error w.r.t. Q function weights: dL/dWq = dL/dq * dQ/dWq
        dL_dWq = tf.gradients(
            self.q,
            self.net_weights,
            -dL_dq,
        )

        # gradient of Bellman Error w.r.t. Q function action input: dL/da = dL/dq * dq/da
        dL_da = tf.gradients(
            self.q,
            self.act_in,
            -dL_dq,
        )

        # gradient of Bellman Error w.r.t. Q function obs input: dL/do = dL/dq * dq/do
        dL_do = tf.gradients(
            self.q, 
            self.obs_in,
            -dL_dq,
        )
        # obs extractor is not recurrent, so sum up grads from all timesteps
        dL_do = tf.reduce_sum(dL_do, axis=2)

        return dL_dq, dL_dWq, dL_da, dL_do

    def apply_gradients_to_Q_net(self,
            dL_dq,
            act,
            obs,
            num_step=1,
        ):
        """
        compute and apply gradient of Bellman Error w.r.t. Q network weights

        inputs:
            dL_da: gradient of loss with respect to Q value, numpy array, 
                shape (N, num_timestep, 1)
            act: series of actions, numpy array,
                shape (N, num_timestep, act_dim)
            obs: series of observations, numpy array, 
                shape (N, num_timestep, obs_dim)
            num_step: number of gradient steps to perform
        """
        for i in range(0, num_step):on available runtime hardware and constraints, this layer will choose different implementations (cuDNN-based or pure-TensorFlow) to maximize the performance. If a GPU is available and all the arguments to the layer meet the requirement of the CuDNN kernel (see below for details), the layer will use a fast cuDNN implementation.
            self.sess.run(
                self.grad_step,
                feed_dict={
                    self.dL_dq: dL_dq,
                    self.act_in: act,
                    self.obs_in: obs,
                }
            )
        return


if __name__ == "__main__":

    session = tf.compat.v1.Session()

    critic = CriticRDPG(
        session,
        training=True,
    )
    critic.print_network_info()

    # test feature extraction forward pass
    # make a stack of unique observations:
    obs = []
    for i in range(0, 4):
        img = np.random.randn(1, 16, 90, 3)
        o = critic.sample_obs(img)
        print(o.shape)
        obs.append(o)
    obs = np.array([obs])
    print('\nobs shape:\t', obs.shape)

    # test dL_da
    dL_dq = np.random.randn(1, 4, 1)
    act = np.random.randn(1, 4, 3)
    dL_da = critic.get_dL_da(dL_dq, obs, act)
    print('\ndL/da:\n', dL_da)

    # test dL_do
    dL_do = critic.get_dL_do(dL_dq, obs, act)
    print('\ndL/do:\n', dL_do)

    # test q
    q = critic.sample_q(obs, act)
    print('\nq:\n', q)
    q_t = critic.sample_q_target(obs, act)
    print('\nq\':\n', q_t)

    # test gradient update to Q network
    critic.apply_gradients_to_Q_net(dL_dq, act, obs)
    pass

