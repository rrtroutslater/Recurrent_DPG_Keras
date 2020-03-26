from __future__ import print_function
import tensorflow as tf 
from tensorflow import keras
import numpy as np


class Encoder():
    def __init__(self,
            session,
            img_dim=[16,90,3], 
            obs_dim=32,
            learning_rate=0.005,
            beta = 1.0,
            test_mode=False,
        ):
        tf.compat.v1.disable_eager_execution()

        self.sess = session        
        self.img_dim = img_dim
        self.obs_dim = obs_dim
        self.learning_rate = learning_rate
        self.test_mode = test_mode
        
        # mixing factor for gradients of Bellman error w.r.t. feature
        # dL/dWf = (dL/dfa + b * dL_dfQ) * df_dWf
        self.beta = beta 

        # encoder, takes stack of range images, returns extracted features
        self.net, self.img_in, self.obs = self.make_encoder_net()
        self.net_weights = self.net.trainable_weights

        # gradients
        # self.dL_dfQ, self.dL_dfa, self.dL_dWf = self.initialize_gradients()
        self.initialize_gradients()

        # gradient step
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.grad_step = self.optimizer.apply_gradients(zip(self.dL_dWf, self.net_weights))

        self.sess.run(tf.compat.v1.global_variables_initializer())
        return

    def sample_obs(self,
            img_in,
        ):
        # obs = self.net.predict(img_in)
        obs = self.sess.run(
            self.obs,
            feed_dict={
                self.img_in: img_in
            }
        )
        return np.expand_dims(obs, axis=0)

    def initialize_gradients(self,):
        """
        """
        # placeholder for gradient of Bellman Error w.r.t. feature from Q function
        self.dL_dfQ = tf.keras.backend.placeholder(
            # shape=[None, None, self.obs_dim],
            shape=[None, self.obs_dim],
            dtype=tf.float32,
        )

        # placeholder for gradient of Bellman Error w.r.t. feature from actor
        self.dL_dfa = tf.keras.backend.placeholder(
            # shape=[None, None, self.obs_dim],
            shape=[None, self.obs_dim],
            dtype=tf.float32,
        )

        print(self.dL_dfa.get_shape())
        print(self.dL_dfQ.get_shape())


        # gradient of Bellman Error w.r.t. feature extractor weights: dL/dWf = (dL/dfa + b * dL_dfQ) * df_dWf
        self.dL_dWf = tf.gradients(
            self.obs, 
            self.net_weights,
            -(self.dL_dfa + self.beta * self.dL_dfQ),
        )

        return
        # return dL_dfQ, dL_dfa, dL_dWf

    def apply_gradients_to_feature_extractor(self,
            dL_dfQ,
            dL_dfa,
            img_in,
            num_step
        ):
        """
        """
        for i in range(0, num_step):
            if self.test_mode:
                print('----------\nweights before update:', self.net_weights[0].eval(session=self.sess))
            self.sess.run(
                self.grad_step,
                feed_dict={
                    self.dL_dfQ: dL_dfQ,
                    self.dL_dfa: dL_dfa,
                    self.img_in: img_in,
                }
            )
            if self.test_mode:
                print('----------\nweights after update:', self.net_weights[0].eval(session=self.sess))
        return

    def make_encoder_net(self):
        """
        encoder network
        [batch_size,16,90,3] -> [batch_size,64]
        """
        img_in = keras.layers.Input(shape=self.img_dim)

        conv_1 = keras.layers.Conv2D(
            filters=2,
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(img_in)
        conv_2 = keras.layers.Conv2D(
            filters=4,
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_1)
        conv_3 = keras.layers.Conv2D(
            filters=8,
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_2)
        conv_4 = keras.layers.Conv2D(
            filters=16,
            kernel_size=[3,3],
            strides=[2,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_3)
        conv_5 = keras.layers.Conv2D(
            filters=32,
            kernel_size=[3,3],
            strides=[1,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_4)
        conv_6 = keras.layers.Conv2D(
            filters=64,
            kernel_size=[3,3],
            strides=[1,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_5)

        flat = keras.layers.Flatten()(conv_6)

        feature = keras.layers.Dense(self.obs_dim)(flat)

        if self.test_mode:
            print('obs shape:\t', img_in.get_shape())
            print('conv_1 shape:\t', conv_1.get_shape())
            print('conv_2 shape:\t', conv_2.get_shape())
            print('conv_3 shape:\t', conv_3.get_shape())
            print('conv_4 shape:\t', conv_4.get_shape())
            print('conv_5 shape:\t', conv_5.get_shape())
            print('conv_6 shape:\t', conv_6.get_shape())
            print('dense shape:\t', flat.get_shape())
            print('feature shape:\t', feature.get_shape())

        extractor = keras.Model(img_in, feature)
        return extractor, img_in, feature

if __name__ == "__main__":

    session = tf.compat.v1.Session()
    encoder = Encoder(session, test_mode=True)

    # test forward pass
    img = np.random.randn(1, 16, 90, 3)
    obs = encoder.sample_obs(img)
    print('\nobs:\n', obs)
    print('\nobs shape:\n', obs.shape)

    # test gradient
    # dL_do = np.random.randn(10,32)
    # dJ_da = np.random.randn(10,32)
    # img_in = np.random.randn(10, 16, 90 ,3)
    # encoder.apply_gradients_to_feature_extractor(dL_do, dJ_da, img_in, 1)

    pass