from __future__ import print_function
from ACBase import *
from EncoderNet import *
import math

class ActorRDPG(ACBase):
    def __init__(self,
                 session,
                 obs_dim=[16, 90, 3],
                #  obs_dim=32,
                 act_dim=3,
                 learning_rate=0.005,
                 training=True,
                 test_mode=False,
                 lstm_horizon=20
                 ):

        tf.compat.v1.disable_eager_execution()
        self.set_network_type("actor")

        self.sess = session
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.learning_rate = learning_rate
        self.training = training
        self.test_mode = test_mode
        self.lstm_horizon = lstm_horizon

        self.lstm_units = 16 # number of units in LSTM cell

        # hidden and carry state numpy arrays
        self.h_prev = np.random.randn(1, self.lstm_units)
        self.c_prev = np.random.randn(1, self.lstm_units)
        self.h_prev_t = np.random.randn(1, self.lstm_units)
        self.c_prev_t = np.random.randn(1, self.lstm_units)

        # actor, mu(o)
        self.net, self.obs_in, self.act, self.actor_h_sequence, \
            self.actor_h, self.actor_c, \
            self.actor_h_ph, self.actor_c_ph \
            = self.make_act_net('normal')
        self.net_weights = self.net.trainable_weights

        # actor target, mu'(o)
        self.net_t, self.obs_t_in, self.act_t, self.actor_h_t_sequence, \
            self.actor_h_t, self.actor_c_t, \
            self.actor_h_t_ph, self.actor_c_t_ph \
            = self.make_act_net('target')
        self.net_t_weights = self.net_t.trainable_weights

        # gradients
        self.dQ_da, self.dQ_dWa = self.initialize_gradients()
        self.grads = {
            "dQ/da": self.dQ_da,
            "dQ/dWa = 1/NT (dQ/da * dmu/dWa)": self.dQ_dWa,
        }

        # gradient step
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.grad_step = self.optimizer.apply_gradients(
            zip(self.dQ_dWa, self.net_weights)
        )

        self.sess.run(tf.compat.v1.global_variables_initializer())
        return

    def make_act_net(self, net_type):
        """
        define actor network architecture, used for actor and actor target.
        encoder -> LSTM

        inputs:
            net type, string to differentiate target/non-target hidden/carry variables

        returns:
            keras model 
            input layer
            action output
            hidden state history h[]
            hidden state h output
            carry state c output
            hidden state h keras variable for manual handling
            carry state c keras variable for manual handling
        """
        # variables for tracking hidden state over variable-length episodes
        # h_ph = tf.keras.backend.variable(self.h_prev, name="h"+net_type)
        # c_ph = tf.keras.backend.variable(self.c_prev, name="c"+net_type)
        h_ph = tf.keras.backend.placeholder(shape=[1, self.lstm_units], name="h"+net_type)
        c_ph = tf.keras.backend.placeholder(shape=[1, self.lstm_units], name="c"+net_type)

        obs_in = keras.layers.Input(
            shape=[None, self.obs_dim[0], self.obs_dim[1], self.obs_dim[2]],
        )

        feature = make_encoder_net(obs_in, test_mode=self.test_mode)
        feature = tf.expand_dims(feature, axis=0)

        lstm_sequence, h, c = tf.keras.layers.LSTM(
            self.lstm_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            bias_initializer='zeros',
            return_sequences=True,
            return_state=True,
            stateful=False,
        )(feature, initial_state=[h_ph, c_ph])

        act = keras.layers.Dense(
            units=self.act_dim,
            activation="tanh",
        )(lstm_sequence)

        model = keras.Model(inputs=[obs_in], outputs=[act, h, c])
        return model, obs_in, act, lstm_sequence, h, c, h_ph, c_ph

    def sample_act(self,
                   obs,
                   add_noise=False,
                   reset_hidden_after_sample=False
                   ):
        """
        sampe action, a = mu(obs)

        input:
            obs: observation, numpy array of shape (N, num_timestep, obs_dim)
            add_noise: flag indicating whether to add noise to action, used for exploration
                of state/action space during training
            reset_hidden_after_sample: indicates whether to reset hidden state to initial
            value after forward propagation, used to ensure grad calcs are based on accurate
            initial hidden state during training

        returns:
            action, numpy array of shape (N, num_timestep, act_dim)
        """
        if reset_hidden_after_sample:
            h_init = self.h_prev
            c_init = self.c_prev

        if self.test_mode:
            print('\n------------------\nbefore forward pass')
            self.display_hidden_state()

        act, h_prev, c_prev = self.sess.run(
            [self.act, self.actor_h, self.actor_c],
            feed_dict={
                self.obs_in: obs,
                self.actor_h_ph: self.h_prev,
                self.actor_c_ph: self.c_prev,
            }
        )

        if self.test_mode:
            # get entire hidden state history
            hidden_state_history = []
            hidden_state_history = self.sess.run(
                [self.actor_h_sequence],
                feed_dict={
                    self.obs_in: obs,
                    self.actor_h_ph: self.h_prev,
                    self.actor_c_ph: self.c_prev,
                }
            )

        # store the final hidden states from time t to be used as inital hidden states at time t+1
        self.h_prev = h_prev
        self.c_prev = c_prev

        if reset_hidden_after_sample:
            self.h_prev = h_init
            self.c_prev = c_init

        if self.test_mode:
            print('\n------------------\nafter forward pass')
            print('act:\n', act)
            self.display_hidden_state()
            print('\nh history:\n', hidden_state_history)

        return act

    def sample_act_target(self,
                          obs,
                          reset_hidden_after_sample=False
                          ):
        """
        sample action from target network, a' = mu'(obs')

        input:input shapes
            obs: observation, numpy array of shape (N, num_timestep, obs_dim)\
            reset_hidden_after_sample: indicates whether to reset hidden state to initial
            value after forward propagation, used to ensure grad calcs are based on accurate
            initial hidden state during training

        returns:
            action, numpy array of shape (N, num_timestep, act_dim)
        """
        if reset_hidden_after_sample:
            h_init = self.h_prev_t
            c_init = self.c_prev_t

        if self.test_mode:
            print('\n-----------------------------\nbefore forward pass')
            self.display_target_hidden_state()

        act_t, h_prev_t, c_prev_t = self.sess.run(
            [self.act_t, self.actor_h_t, self.actor_c_t],
            feed_dict={
                self.obs_t_in: obs,
                self.actor_h_t_ph: self.h_prev_t,
                self.actor_c_t_ph: self.c_prev_t,
            }
        )

        # get entire hidden state history, for testing
        if self.test_mode:
            hidden_state_history = []
            hidden_state_history = self.sess.run(
                [self.actor_h_t_sequence],
                feed_dict={
                    self.obs_t_in: obs,
                    self.actor_h_t_ph: self.h_prev_t,
                    self.actor_c_t_ph: self.c_prev_t,
                }
            )

        # store the final hidden states from time t to be used as inital hidden states at time t+1
        self.h_prev_t = h_prev_t
        self.c_prev_t = c_prev_t

        if reset_hidden_after_sample:
            self.h_prev_t = h_init
            self.c_prev_t = c_init

        if self.test_mode:
            print('\n-----------------------------\nafter forward pass')
            print('\nact TARGET:\n', act_t)
            self.display_target_hidden_state()
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
            placeholder for upstream gradient of RL objective w.r.t. action: dQ/da
            gradient of objective w.r.t. network weights, a list
        """
        # placeholder for result of backpropagation of objective gradient through Critic Q(s,a)
        dQ_da = tf.keras.backend.placeholder(
            shape=[None, None, self.act_dim],
            dtype=tf.float32,
        )

        # list of gradients of objective w.r.t. actor (mu) weights: dQ/dWa = dmu/dWa * dQ/da, mu(f) = a.
        # comes from objective maximization via policy gradient.
        dQ_dWa = tf.gradients(
            self.act,
            self.net_weights,
            -dQ_da,
        )

        return dQ_da, dQ_dWa

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

        for i in range(0, num_step):
            if self.test_mode:
                print('----------\ninput shapes:')
                print('obs shape:\t', obs.shape)
                print('dQ/da shape:\t', dQ_da.shape)
                print('act shape:\t', self.act.get_shape())
                print('\nweights before update:',
                      self.net_weights[0].eval(session=self.sess))
                self.display_hidden_state()

            # grad calculation
            dQ_dWa = self.sess.run(
                self.dQ_dWa,
                feed_dict={
                    self.obs_in: obs,
                    self.dQ_da: dQ_da,
                    self.actor_h_ph: self.h_prev,
                    self.actor_c_ph: self.c_prev,
                }
            )

            # divide by batch size * time length
            for i in range(0, len(dQ_dWa)):
                dQ_dWa[i] /= (obs[0].shape[0] * obs[0].shape[1])


            # application of gradient
            _ = self.sess.run(
                [self.grad_step],
                feed_dict={
                    self.obs_in: obs,
                    self.dQ_da: dQ_da,
                    self.actor_h_ph: self.h_prev,
                    self.actor_c_ph: self.c_prev,
                }
            )

            if self.test_mode:
                print('\nweights after update:',
                      self.net_weights[0].eval(session=self.sess))
                for i in range(0, len(dQ_dWa)):
                    print(dQ_dWa[i].shape)
                self.display_hidden_state()

        return

    def model_predict(self, obs):

        print('h before:\n', self.h_prev)
        print('c before:\n', self.c_prev)
        act, h, c = self.net.predict(obs)
        print(act)
        print(h)
        print(c)
        print('h after:\n', self.h_prev)
        print('c after:\n', self.c_prev)
        return


if __name__ == "__main__":

    session = tf.compat.v1.Session()

    actor = ActorRDPG(
        session,
        training=True,
        test_mode=True,
    )
    actor.export_model_figure()

    # test model saving and loading
    if 1:
        print(type(actor.net))
        actor.net.save_weights('egg_mpode', save_format='tf')
        actor.net.load_weights('egg_mpode')

    # test forward pass with model.predict
    if 0:
        lstm_horizon = 4
        np.random.seed(0)
        o = np.random.randn(1, lstm_horizon, 16, 90, 3)
        actor.model_predict(o)

    # test forward pass
    if 1:
        lstm_horizon = 4
        # np.random.seed(0)
        o = np.random.randn(1, lstm_horizon, 16, 90, 3)
        a = actor.sample_act(o)
        actor.display_hidden_state()
        o1 = np.random.randn(1, 1, 16, 90, 3)
        o2 = np.random.randn(1, 1, 16, 90, 3)
 
        print('===================================================')
        a1 = actor.sample_act(o1)
        print('===================================================')
        a1 = actor.sample_act(o2)
        print('===================================================')
        a1 = actor.sample_act(o1, reset_hidden_after_sample=True)

    # test target forward pass
    if 0: 
        print('===================================================')
        o1 = np.random.randn(1, 1, 16, 90, 3)
        a1 = actor.sample_act_target(o1)
        print(a1)
        print('===================================================')
        a1 = actor.sample_act_target(o1)
        print(a1)

    # test gradient update
    if 0:
        lstm_horizon = 4
        np.random.seed(0)
        o = np.random.randn(1, lstm_horizon, 16, 90, 3)
        dQ_da = np.random.randn(1, lstm_horizon, 3)
        actor.apply_gradients(o, dQ_da, num_step=1)

    # test gradient update AFTER forward propagation ~ this is what happens during training
    if 0: 
        lstm_horizon = 4
        np.random.seed(0)
        o = np.random.randn(lstm_horizon, 16, 90, 3)
        a = actor.sample_act(o)

        o1 = np.random.randn(2, 16, 90, 3)
        dQ_da = np.random.randn(lstm_horizon, 3)
        actor.apply_gradients(o1, dQ_da, num_step=1)

    pass

