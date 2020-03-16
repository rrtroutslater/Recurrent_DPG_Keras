from ACBase import *

class ActorRDPG(ACBase):
    def __init__ (self,
            session,
            obs_dim=32,
            act_dim=3,
            lstm_horizon=4,
            # training=False,
            learning_rate=0.005,
            training=True,
        ):

        tf.compat.v1.disable_eager_execution()
        self.set_network_type("actor")
        
        self.sess = session
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if training:
            self.lstm_horizon = lstm_horizon
        else:
            self.lstm_horizon = 1
        self.training = training
        self.learning_rate = learning_rate

        # actor, mu(o)
        self.net, self.obs_in, self.act, self.actor_h, self.actor_c = self.make_act_net()
        self.net_weights = self.net.trainable_weights

        # actor target, mu'(o)
        self.net_t, self.obs_t_in, self.act_t, self.actor_h_t, self.actor_c_t = self.make_act_net()
        self.net_t_weights = self.net_t.trainable_weights

        # gradients
        self.dL_da, self.dL_dWa, self.dL_do = self.initialize_gradients()
        self.grads = {
            "dL/da": self.dL_da,
            "dL/dWa = dL/da * dmu/dWa": self.dL_dWa,
            "dL/df = dL/da * dmu/df": self.dL_do,
        }

        # gradient step
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.grad_step = self.optimizer.apply_gradients(zip(self.dL_dWa, self.net_weights))

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
            shape=[self.lstm_horizon, self.obs_dim]
        )

        lstm_out, h, c = keras.layers.LSTM(
            units=16,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            return_state=True, 
            stateful=False,
        )(obs_in)

        act = keras.layers.Dense(
            units=self.act_dim,
            activation="tanh",
        )(lstm_out)

        model = keras.Model(inputs=obs_in, outputs=act)
        return model, obs_in, act, lstm_out, c

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

        h = self.sess.run(
            self.actor_h,
            feed_dict={
                self.obs_in: obs
            }
        )
        print('\nh sequence:\n', h)


        act = self.net.predict(obs)
        if add_noise:
            noise = np.random.randn(act.shape)
            act += noise
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
        act_t = self.net_t.predict(obs)
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
            placeholder for upstream gradient of Bellman Loss w.r.t. action: dL/da
            gradient of Bellman Loss w.r.t. network weights, a list
            gradient of Bellman Loss w.r.t. observation input
        """
        # placeholder for result of backpropagation of Bellman Error through Critic Q(s,a)
        dL_da = tf.keras.backend.placeholder(
            shape=[None, self.lstm_horizon, self.act_dim],
            dtype=tf.float32,
        )

        # list of gradients of Bellman Error w.r.t. actor (mu) weights: dL/dWa = dmu/dWa * dL/da, mu(f) = a
        dL_dWa = tf.gradients(
            self.act,
            self.net_weights,
            -dL_da,
        )

        # gradient of Bellman Error w.r.t. obs input: dL/do = dmu/do * dL/da, mu(f) = a
        dL_do = tf.gradients(
            self.act,
            self.obs_in,
            -dL_da,
        )
        # obs extractor is not recurrent, so sum up grads from all timesteps
        dL_do = tf.reduce_sum(dL_do, axis=2)

        return dL_da, dL_dWa, dL_do

    def apply_gradients(self,
            obs,
            dL_da,
            num_step=1,
        ):
        """
        compute and apply gradient of Bellman Error w.r.t. network weights

        inputs:
            obs: series of observations, numpy array, 
                shape (N, num_timestep, obs_dim)
            dL_da: gradient of loss with respect to actor output, numpy array, 
                shape (N, num_timestep, act_dim)
            num_step: number of gradient steps to perform
        """
        for i in range(0, num_step):
            self.sess.run(
                [self.grad_step],
                feed_dict={
                    self.obs_in: obs,
                    self.dL_da: dL_da,
                }
            )
        return


if __name__ == "__main__":
    session = tf.compat.v1.Session()
    actor = ActorRDPG(
        session,
        training=True,
    )

    np.random.seed(0)
    obs = np.random.randn(1, 4, 32)
    act = actor.sample_act(obs)
    print('\nact:\n', act)

    # TODO test the gradients

    pass
