from __future__ import print_function
from ACBase import *
import matplotlib.pyplot as plt
import h5py

class CriticRDPG(ACBase):
    def __init__(self,
                 session,
                #  obs_dim=[16, 90, 3],
                 obs_dim=192,
                 act_dim=3,
                 learning_rate=0.001,
                 training=True,
                 test_mode=False,
                 lstm_units=32,
                 lstm_horizon=20,
                 critic_fn="",
                 critic_target_fn="",
                 ):

        tf.compat.v1.disable_eager_execution()
        self.set_network_type("critic")

        self.sess = session
        # self.sess = tf.compat.v1.Session()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.training = training
        self.learning_rate = learning_rate
        self.test_mode = test_mode
        self.lstm_units = lstm_units
        self.lstm_horizon = lstm_horizon

        # hidden and carry state numpy arrays
        self.reset_hidden_states()

        # critic, Q(o, mu(o))
        self.net, self.act_in, self.obs_in, self.q, self.critic_h_sequence, \
            self.critic_h, self.critic_c, \
            self.critic_h_ph, self.critic_c_ph \
            = self.make_Q_net('normal')
        self.net_weights = self.net.trainable_weights

        # loss, used for grad calcs
        self.y_ph = tf.keras.backend.placeholder(shape=[None, None, 1])
        mse = tf.keras.losses.MeanSquaredError()
        self.loss = mse(self.y_ph, self.q)

        # critic target, Q'(o', mu(o'))
        self.net_t, self.act_in_t, self.obs_in_t,  self.q_t, self.critic_h_t_sequence, \
            self.critic_h_t, self.critic_c_t, \
            self.critic_h_t_ph, self.critic_c_t_ph \
            = self.make_Q_net('target')
        self.net_t_weights = self.net_t.trainable_weights

        # gradients
        self.dL_dWq, self.dQ_da = self.initialize_gradients()
        self.grads = {
            "dL/dWq = dL/dq * dQ/dWq": self.dL_dWq,
            "dQ/da": self.dQ_da,
        }

        # optimizer, grad step
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.grad_step = self.optimizer.apply_gradients(
            zip(self.dL_dWq, self.net_weights),
        )
        return

    def reset_hidden_states(self,):
        self.h_prev = np.zeros(shape=(1, self.lstm_units))
        self.c_prev = np.zeros(shape=(1, self.lstm_units))
        self.h_prev_t = np.zeros(shape=(1, self.lstm_units))
        self.c_prev_t = np.zeros(shape=(1, self.lstm_units))
        return

    def get_loss(self, act, obs, y):
        loss = self.sess.run(
            self.loss,
            feed_dict={
                self.y_ph: y,
                self.act_in: act,
                self.obs_in: obs,
                self.critic_h_ph: self.h_prev,
                self.critic_c_ph: self.c_prev,
            }
        )
        return loss

    def make_Q_net(self, net_type):
        """
        define critic network architecture, used for Q and Q target.
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
        # placeholders for tracking hidden state over variable-length episodes
        h_ph = tf.keras.backend.placeholder(shape=[1, self.lstm_units], name="h_"+net_type)
        c_ph = tf.keras.backend.placeholder(shape=[1, self.lstm_units], name="c_"+net_type)


        act_in = tf.keras.layers.Input(shape=[None, self.act_dim], name="act_in")
        obs_in = tf.keras.layers.Input(shape=[None, self.obs_dim], name="obs_in")

        obs_desnse = keras.layers.Dense(
            units=48,
            activation='relu',
            name="Q_obs_expand_"+net_type,
        )(obs_in)
        feature = tf.expand_dims(obs_desnse, axis=0)

        act_expanded = keras.layers.Dense(
            units=16,
            activation="relu",
            # activation=None,
            name="Q_act_expand"+net_type,
        )(act_in)
        act_obs_in = tf.concat([act_expanded, obs_in], axis=2)

        lstm_sequence, h, c = keras.layers.LSTM(
            units=self.lstm_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            return_state=True,
            stateful=False,
            name="lstm_critic_"+net_type,
            # recurrent_dropout=0.02,
            # dropout=0.02,
        )(act_obs_in, initial_state=[h_ph, c_ph])

        pre_q = keras.layers.Dense(
            units=8,
            activation=None,
            name="Q_pre_"+net_type,
        )(lstm_sequence)

        q = keras.layers.Dense(
            units=1,
            # activation='relu',
            activation=keras.layers.LeakyReLU(alpha=0.3),
            name="Q_"+net_type,
        # )(lstm_sequence)
        )(pre_q)

        model = keras.Model(inputs=[act_in, obs_in], outputs=[q, h, c])
        return model, act_in, obs_in, q, lstm_sequence, h, c, h_ph, c_ph

    def sample_q(self,
                 obs,
                 act,
                 reset_hidden_after_sample=False
                 ):
        """
        sample Q(o, a) value

        inputs: 
            obs: observations, numpy array of shape (N, num_timestep, obs_dim)
            act: actions, numpy array of shape (N, num_timestep, act_dim)
            reset_hidden_after_sample: indicates whether to reset hidden state to initial
            value after forward propagation, used to ensure grad calcs are based on accurate
            initial hidden state during training

        returns:
            q, numpy array of shape (N, num_timestep, 1)
        """
        if reset_hidden_after_sample:
            h_init = self.h_prev
            c_init = self.c_prev

        if self.test_mode:
            print('\n-----------------------------\nbefore forward pass')
            self.display_hidden_state()

        q, h_prev, c_prev = self.sess.run(
            [self.q, self.critic_h, self.critic_c],
            feed_dict={
                self.obs_in: obs,
                self.act_in: act,
                self.critic_h_ph: self.h_prev,
                self.critic_c_ph: self.c_prev,
            }
        )

        if self.test_mode:
            # get entire hidden state history
            hidden_state_history = self.sess.run(
                [self.critic_h_sequence],
                feed_dict={
                    self.obs_in: obs,
                    self.act_in: act,
                    self.critic_h_ph: self.h_prev,
                    self.critic_c_ph: self.c_prev,
                }
            )

        # store the final hidden states from time t to be used as inital hidden states at time t+1
        self.h_prev = h_prev
        self.c_prev = c_prev

        if reset_hidden_after_sample:
            self.h_prev = h_init
            self.c_prev = c_init

        if self.test_mode:
            print('\n-----------------------------\nafter forward pass')
            print('\nq:\n', q)
            self.display_hidden_state()
            print('\nhidden state history:\n', hidden_state_history)

        return q

    def sample_q_target(self,
                        obs,
                        act,
                        reset_hidden_after_sample=False
                        ):
        """
        sample Q'(o', a') value from TARGET network

        inputs: 
            obs: observations, numpy array of shape (N, num_timestep, obs_dim)
            act: actions, numpy array of shape (N, num_timestep, act_dim)
            reset_hidden_after_sample: indicates whether to reset hidden state to initial
            value after forward propagation, used to ensure grad calcs are based on accurate
            initial hidden state during training

        returns:
            q, numpy array of shape (N, num_timestep, 1)
        """
        if reset_hidden_after_sample:
            h_init = self.h_prev_t
            c_init = self.c_prev_t

        if self.test_mode:
            print('\n-----------------------------\nbefore forward pass')
            self.display_target_hidden_state()

        q_t, h_prev_t, c_prev_t = self.sess.run(
            [self.q_t, self.critic_h_t, self.critic_c_t],
            feed_dict={
                self.obs_in_t: obs,
                self.act_in_t: act,
                self.critic_h_t_ph: self.h_prev_t,
                self.critic_c_t_ph: self.c_prev_t,
            }
        )

        if self.test_mode:
            # get entire hidden state history
            hidden_state_history = self.sess.run(
                [self.critic_h_t_sequence],
                feed_dict={
                    self.obs_in_t: obs,
                    self.act_in_t: act,
                    self.critic_h_t_ph: self.h_prev_t,
                    self.critic_c_t_ph: self.c_prev_t,
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
            print('\nq:\n', q_t)
            self.display_target_hidden_state()
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
            dQ/da, gradient of objective w.r.t. action
        """
        dQ_da = self.sess.run(
            self.dQ_da,
            feed_dict={
                self.obs_in: obs,
                self.act_in: act,
                self.critic_h_ph: self.h_prev,
                self.critic_c_ph: self.c_prev,
            }
        )
        return dQ_da[0]

    def initialize_gradients(self,):
        """
        initialize gradients of Q(o, a) w.r.t. obs and action

        returns:
            placeholder for upstream gradient of Bellman Error w.r.t. Q function output
            gradient of Bellman Error w.r.t. Q network weights
            gradient of Bellman Error w.r.t. action input 
                backpropagated through actor and feature extractor during training.eval(session=self.sess)[0]
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
        return dL_dWq, dQ_da

    def train_critic(self, act, obs, y, num_step=1):
        """
        take a gradient step on critic function
        """
        losses = []
        # print('WEIGHTS in train critic BEFORE:\n', self.net_weights[0].eval(self.sess)[0][0)

        _ = self.sess.run(
            [self.grad_step],
            feed_dict={
                self.y_ph: y,
                self.act_in: act,
                self.obs_in: obs,
                self.critic_h_ph: self.h_prev,
                self.critic_c_ph: self.c_prev,
            }
        )

        loss = self.sess.run(
            [self.loss],
            feed_dict={
                self.y_ph: y,
                self.act_in: act,
                self.obs_in: obs,
                self.critic_h_ph: self.h_prev,
                self.critic_c_ph: self.c_prev,
            }
        )
        losses.append(loss[0])
        # print('WEIGHTS in train critic AFTER:\n', self.net_weights[0].eval(self.sess)[0][0][0])
        return losses


if __name__ == "__main__":
    session = tf.compat.v1.Session()
    critic = CriticRDPG(
        session,
        training=True,
        test_mode=True,
    )

    # test forward pass on series of single observations
    if 0:
        for i in range(0, 4):
            print('\n--------------')
            obs = np.random.randn(1, 1, 16, 90, 3)
            act = np.random.randn(1, 1, 3)
            q = critic.sample_q(obs, act)

    # test model saving
    if 1:
        net_fn, target_fn = critic.save_model()
        critic.load_model(net_fn, target_fn)

    # make a stack of unique observations:
    obs = np.random.randn(4, 16, 90, 3)
    act = np.random.randn(4, 3)

    # test forward pass on batch with hidden state propagation
    if 0:
        # target
        print("\ncritic target:\n")
        q = critic.sample_q_target(obs, act)
        q_1 = critic.sample_q_target(np.random.randn(
            1, 16, 90, 3), np.random.randn(1, 3))

        print("\ncritic:\n")
        # actor
        q = critic.sample_q(obs, act)
        q_1 = critic.sample_q(np.random.randn(
            1, 16, 90, 3), np.random.randn(1, 3))

    # test gradients
    if 0:
        # test dQ_da
        act = np.random.randn(4, 3)
        dQ_da = critic.get_dQ_da_critic(obs, act)
        print('\ndQ/da:\n', dQ_da)

    if 0:
        # test q
        act = np.random.randn(4, 3)
        q = critic.sample_q(obs, act)
        print('\nq:\n', q)
        q_t = critic.sample_q_target(obs, act)
        print('\nq\':\n', q_t)

    # test training
    if 0:
        # forward-propagate "prior to episode" data
        q = critic.sample_q(obs, act)

        act = np.random.randn(4, 3)
        obs = np.random.randn(4, 16, 90, 3)
        y = np.random.randn(4, 1)
        losses = []
        loss = critic.train_critic(act, obs, y)
        
        plt.plot(loss)
        plt.show()
        # print(loss)
    pass
