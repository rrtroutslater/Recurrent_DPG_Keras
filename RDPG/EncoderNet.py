from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np


def make_encoder_net(img_in, feature_dim=32, test_mode=False):
    """
    simple encoder architecture
    """
    conv_1 = keras.layers.Conv2D(
        filters=2,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
    )(img_in)
    conv_2 = keras.layers.Conv2D(
        filters=4,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
    )(conv_1)
    conv_3 = keras.layers.Conv2D(
        filters=8,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
    )(conv_2)
    conv_4 = keras.layers.Conv2D(
        filters=16,
        kernel_size=[3, 3],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
    )(conv_3)
    conv_5 = keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        strides=[1, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
    )(conv_4)
    conv_6 = keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
    )(conv_5)

    flat = keras.layers.Flatten()(conv_6)

    feature = keras.layers.Dense(feature_dim)(flat)

    if test_mode:
        print('obs shape:\t', img_in.get_shape())
        print('conv_1 shape:\t', conv_1.get_shape())
        print('conv_2 shape:\t', conv_2.get_shape())
        print('conv_3 shape:\t', conv_3.get_shape())
        print('conv_4 shape:\t', conv_4.get_shape())
        print('conv_5 shape:\t', conv_5.get_shape())
        print('conv_6 shape:\t', conv_6.get_shape())
        print('dense shape:\t', flat.get_shape())
        print('feature shape:\t', feature.get_shape())

    return feature
