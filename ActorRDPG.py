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
            training=False,
        ):

        # session = tf.compat.v1.Session()
        # keras.backend.set_session(session)

        self.feature_dim = feature_dim
        self.act_dim = act_dim
        if training:
            self.lstm_horizon = lstm_horizon
        else:
            self.lstm_horizon = 1

        # actor, mu(o)
        self.actor, self.actor_h, self.actor_c = self.make_act_net()
        self.actor_weights = self.actor.trainable_weights

        # actor target, mu'(o)
        self.actor_t, self.actor_h_t, self.actor_c_t = self.make_act_net()
        self.actor_weights_t = self.actor_t.trainable_weights


        
        return

    def make_act_net(self):
        """
        makes an actor network

        returns keras model object
        """

        net_in = keras.layers.Input(
            shape=[self.lstm_horizon, self.feature_dim]
        )
        print(net_in.get_shape())

        lstm_out, h, c = keras.layers.LSTM(
            units=16,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            return_state=True, 
            stateful=False,
        )(net_in)
        print(lstm_out.get_shape())

        act = keras.layers.Dense(
            units=self.act_dim,
            activation="tanh",
        )(lstm_out)

        model = keras.Model(inputs=net_in, outputs=act)
        return model, h, c


if __name__ == "__main__":
    pass
