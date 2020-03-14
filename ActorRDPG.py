from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np

class ActorRDPG():
    """
    """
    def __init__ (self,
            feature_dim=64,
            act_dim=3,
            lstm_horizon=4,
            # training=False,
            training=True,
            learning_rate=0.005,
        ):

        self.feature_dim = feature_dim
        self.act_dim = act_dim
        if training:
            self.lstm_horizon = lstm_horizon
        else:
            self.lstm_horizon = 1
        self.learning_rate = learning_rate

        # actor, mu(o)
        self.actor_net, self.actor_h, self.actor_c = self.make_act_net()
        self.actor_weights = self.actor_net.trainable_weights

        # actor target, mu'(o)
        self.actor_net_t, self.actor_h_t, self.actor_c_t = self.make_act_net()
        self.actor_weights_t = self.actor_net_t.trainable_weights

        # gradients
        self.q_grad, self.param_grad, self.input_feature_grad = self.initialize_gradients()

        # gradient step
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.grad_step = self.optimizer.apply_gradients(zip(self.param_grad, self.actor_weights))
        return

    def initialize_gradients(self):
        """
        initialize gradients
        tf.gradients(
            y,  
            x,
            upstream_grads, (eg, dz/dy)
        )
        computes dy/dx * upstream_grads

        returns:
            placeholder for upstream gradient of Q function: dQ/da
            gradient of action w.r.t. weights: da/dW
            gradient of action w.r.t. feature input: da/df
        """
        # placeholder for gradients from Q function (result of backpropagation through Critic)
        q_grad = tf.keras.backend.placeholder(
            shape=[None, self.lstm_horizon, self.act_dim],
            dtype=tf.float32,
        )

        # gradient of actor weights w.r.t. input action, given gradient of q function w.r.t. action
        param_grad = tf.gradients(
            self.actor_net.output,
            self.actor_weights,
            -q_grad,
        )

        # gradient of action w.r.t. feature input, to be backpropagated through feature extractor
        input_feature_grad = tf.gradients(
            self.actor_net.output,
            self.actor_net.input,
            -q_grad,
        )
        # feature extractor is not recurrent, so sum up grads from fall timesteps
        input_feature_grad = tf.reduce_sum(input_feature_grad, axis=2)

        return q_grad, param_grad, input_feature_grad

    def print_network_info(self):
        """
        print model stats, layer shapes, etc
        """
        print("\nactor weights:")
        for w in self.actor_weights:
            print(w)
            print(w.get_shape())

        print("\nactor target weights:")
        for w in self.actor_weights_t:
            print(w)
            print(w.get_shape())

        print("\ngradients:")
        print("dQ/da\n", self.q_grad.get_shape())
        print("da/dW\n", self.param_grad)
        print("da/df\n", self.input_feature_grad.get_shape())
        return

    def export_model_figure(self,
            fnames=["ActorRDPG_architecture.png", "ActorRDPG_target_architecture.png"):
        """
        exports figure of model architecture 
        """
        plot_model(self.actor_net, to_file=fnames[0])
        plot_model(self.actor_net_t, to_file=fnames[1])
        return

    def make_act_net(self):
        """
        makes an actor network

        returns keras model object
        """
        net_in = keras.layers.Input(
            shape=[self.lstm_horizon, self.feature_dim]
        )

        lstm_out, h, c = keras.layers.LSTM(
            units=16,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            return_state=True, 
            stateful=False,
        )(net_in)

        act = keras.layers.Dense(
            units=self.act_dim,
            activation="tanh",
        )(lstm_out)

        model = keras.Model(inputs=net_in, outputs=act)
        return model, h, c

    # def 


if __name__ == "__main__":
    pass
