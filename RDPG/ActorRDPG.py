from __future__ import print_function
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
        self.h_prev = None
        self.c_prev = None
        self.h_prev_t = None
        self.c_prev_t = None

        # actor, mu(o)
        self.net, self.obs_in, self.act, self.actor_h_sequence, self.actor_h, \
            self.actor_c, self.lstm_cell = self.make_act_net()
        self.net_weights = self.net.trainable_weights

        # actor target, mu'(o)
        self.net_t, self.obs_t_in, self.act_t, self.actor_h_t_sequence, self.actor_h_t, \
            self.actor_c_t, _ = self.make_act_net()
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

        lstm_cell = tf.keras.layers.LSTMCell(
            self.lstm_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            bias_initializer='zeros',
        )

        lstm = tf.keras.layers.RNN(
            lstm_cell,
            return_sequences=True,
            return_state=True,
            stateful=False,
        )

        lstm_out, h, c = lstm(obs_in)

        act = keras.layers.Dense(
            units=self.act_dim,
            activation="tanh",
        )(lstm_out)

        model = keras.Model(inputs=obs_in, outputs=act)
        return model, obs_in, act, lstm_out, h, c, lstm_cell

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

        act, h_prev, c_prev = self.sess.run(
            [self.act, self.actor_h, self.actor_c],
            feed_dict={
                self.obs_in: obs,
            }
        )

        if self.test_mode:
            # get entire hidden state history
            hidden_state_history = []
            hidden_state_history = self.sess.run(
                [self.actor_h_sequence],
                feed_dict={
                    self.obs_in: obs,
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

        act_t, h_prev_t, c_prev_t = self.sess.run(
            [self.act_t, self.actor_h_t, self.actor_c_t],
            feed_dict={
                self.obs_t_in: obs,
            }
        )

        # get entire hidden state history, for testing
        if self.test_mode:
            hidden_state_history = []
            hidden_state_history = self.sess.run(
                [self.actor_h_t_sequence],
                feed_dict={
                    self.obs_t_in: obs,
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

        for i in range(0, num_step):
            if self.test_mode:
                print('----------\ninput shapes:')
                print('obs shape:\t', obs.shape)
                print('dQ/da shape:\t', dQ_da.shape)
                print('act shape:\t', self.act.get_shape())
                # print('\nweights before update:',
                #       self.net_weights[0].eval(session=self.sess))

            # grad calculation
            dJ_dWa = self.sess.run(
                self.dJ_dWa,
                feed_dict={
                    self.obs_in: obs,
                    self.dQ_da: dQ_da,
                }
            )

            # application of gradient
            _ = self.sess.run(
                [self.grad_step],
                feed_dict={
                    self.obs_in: obs,
                    self.dQ_da: dQ_da,
                }
            )

            if self.test_mode:
                # print('\ndJ_dWa shape:\t', dJ_dWa.shape)
                # print('\nweights after update:',
                #       self.net_weights[0].eval(session=self.sess))
                for i in range(0, len(dJ_dWa)):
                    print(dJ_dWa[i].shape)
        return

    def get_dJ_do_actor(self,
                        obs,
                        dQ_da,
                        ):
        """
        compute gradient of objective w.r.t. observation. used for backpropagation through 
        encoder.

        inputs:
            obs: observation, numpy array, 
                shape (N, num_timestep, obs_dim)
            dQ_da: gradient of objective w.r.t. action a = mu(o), numpy array, 
                shape (N, num_timestep, act_dim)

        returns:
            dJ/do, gradient of objective w.r.t. observation input
        """
        dJ_do, h, c = self.sess.run(
            [self.dJ_do, self.actor_h, self.actor_c],
            feed_dict={
                self.dQ_da: dQ_da,
                self.obs_in: obs,
            }
        )

        if self.test_mode:
            print("\nhidden states after grad update:")
            print("c:\n", c)
            print("h:\n", h)

        return dJ_do[0]


if __name__ == "__main__":

    session = tf.compat.v1.Session()

    # test gradient update
    if 0:
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

    # test hidden state maintenance without explicit handling of hidden state
    if 0:
        # run pre-training-episode data to set initial hidden states
        N = 1
        lstm_horizon = 4
        actor = ActorRDPG(
            session,
            training=True,
            test_mode=True,
        )
        o = np.random.randn(N, lstm_horizon, 32)
        o1 = np.random.randn(N, 1, 32)

        print("\nactor:\n")
        actor.sample_act(o)
        # now, with "next" sequence, should see initial state = final state of previous
        actor.sample_act(o1)

        print("\nactor target:\n")
        actor.sample_act_target(o)
        # now, with "next" sequence, should see initial state = final state of previous
        actor.sample_act_target(o1)

    # test hidden state maintenance when both grad AND forward prop need to be taken for
    # same set of obs
    if 1:
        N = 1
        lstm_horizon = 4
        actor = ActorRDPG(
            session,
            training=True,
            test_mode=True,
        )
        o = np.random.randn(N, lstm_horizon, 32)
        o1 = np.random.randn(N, 2, 32)
        o2 = np.random.randn(N, 2, 32)
        # dQ_da = np.random.randn(N, 4, 3)
        dQ_da = np.random.randn(N, 2, 3)

        # forward propagate history to set initial hidden states for episode
        print("\nactor:\n")
        actor.sample_act(o)

        # forward propagate an action in the episode
        actor.sample_act(o1)

        # calculate gradient during episode
        # dJ_do = actor.get_dJ_do_actor(o1, dQ_da)

        # apply gradients
        actor.apply_gradients(o1, dQ_da, num_step=1)

        # forward propagate action in the episode. should have same init as final h as above
        actor.sample_act(o2)
        # actor.sample_act(o1) # passing the same obs twice in a row doesn't change h?

        # NOTE: doing gradient calc does NOT change the hidden state.
        # but is it accessing the correct hidden state? appears to.

        # bottom line: grad calcs do not (seem) to alter hidden state, so can calc these
        # before/after forward propagations and don't need to do additional h/c tracking
    pass

"""
backpropagation through time
    - requires summation of:
        dQ/da * dmu/dW

gradients are always returned as arrays of the shame shape as params (regardless of batch
size and time horizon). so assume that the summation of gradients due to each sample and
timestep is automatically done by tf.gradients    
"""
