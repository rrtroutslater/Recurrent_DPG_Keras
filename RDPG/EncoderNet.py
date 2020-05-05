from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np


def make_encoder_net(img_in, feature_dim=48, test_mode=False, name=""):
    """
    simple encoder architecture
    """
    img_in = tf.reduce_sum(img_in, axis=0)

    # img_in = tf.reshape(img_in, [-1, 16, 90, 3])
    # img_in = tf.split(img_in, img_in.get_shape()[1], axis=0)
    # print(type(N))

    # print(img_in.get_shape())
    # img_in = tf.reshape(img_in, shape=[N*T, o1, o2, o3])
    # print(img_in.get_shape())

    # conv_1 = keras.layers.Conv2D(
    #     filters=2,
    #     kernel_size=[7, 7],
    #     strides=[2, 2],
    #     padding='same',
    #     activation=keras.layers.LeakyReLU(alpha=0.3),
    #     name=name+"_1"
    # )(img_in)
    # conv_2 = keras.layers.Conv2D(
    #     filters=4,
    #     kernel_size=[5, 5],
    #     strides=[2, 2],
    #     padding='same',
    #     activation=keras.layers.LeakyReLU(alpha=0.3),
    #     name=name+"_2"
    # )(conv_1)
    # conv_3 = keras.layers.Conv2D(
    #     filters=8,
    #     kernel_size=[5, 5],
    #     strides=[2, 2],
    #     padding='same',
    #     activation=keras.layers.LeakyReLU(alpha=0.3),
    #     name=name+"_3"
    # )(conv_2)
    # conv_4 = keras.layers.Conv2D(
    #     filters=16,
    #     kernel_size=[3, 3],
    #     strides=[2, 2],
    #     padding='same',
    #     activation=keras.layers.LeakyReLU(alpha=0.3),
    #     name=name+"_4",
    # )(conv_3)
    # conv_5 = keras.layers.Conv2D(
    #     filters=32,
    #     kernel_size=[3, 3],
    #     strides=[1, 2],
    #     padding='same',
    #     activation=keras.layers.LeakyReLU(alpha=0.3),
    #     name=name+"_5",
    # )(conv_4)
    # conv_6 = keras.layers.Conv2D(
    #     filters=64,
    #     kernel_size=[3, 3],
    #     strides=[1, 2],
    #     padding='same',
    #     activation=keras.layers.LeakyReLU(alpha=0.3),
    #     name=name+"_6",
    # )(conv_5)

    # flat = keras.layers.Flatten(
    #     name=name+"_conv_flat",
    # )(conv_6)

    # feature = keras.layers.Dense(
    #     feature_dim,
    #     name=name+"_conv_dense",
    # )(flat)

    # if test_mode:
    #     print('obs shape:\t', img_in.get_shape())
    #     print('conv_1 shape:\t', conv_1.get_shape())
    #     print('conv_2 shape:\t', conv_2.get_shape())
    #     print('conv_3 shape:\t', conv_3.get_shape())
    #     print('conv_4 shape:\t', conv_4.get_shape())
    #     print('conv_5 shape:\t', conv_5.get_shape())
    #     print('conv_6 shape:\t', conv_6.get_shape())
    #     print('dense shape:\t', flat.get_shape())
    #     print('feature shape:\t', feature.get_shape())

    conv_1 = keras.layers.Conv2D(
        filters=4,
        kernel_size=[7, 7],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
        name=name+"_1"
    )(img_in)
    conv_2 = keras.layers.Conv2D(
        filters=8,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
        name=name+"_2"
    )(conv_1)
    pool_1 = keras.layers.MaxPool2D(
        pool_size=[1,2],
        strides=[1,2],
        padding='same',
        name=name+"_pool"
    )(conv_2)
    conv_3 = keras.layers.Conv2D(
        filters=16,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
        name=name+"_3"
    )(pool_1)
    conv_4 = keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        strides=[2, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
        name=name+"_4",
    )(conv_3)
    conv_5 = keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        strides=[1, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
        name=name+"_5",
    )(conv_4)
    conv_6 = keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 2],
        padding='same',
        activation=keras.layers.LeakyReLU(alpha=0.3),
        name=name+"_6",
    )(conv_5)
    conv_7 = keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 2],
        padding='same',
        # activation=keras.layers.LeakyReLU(alpha=0.3),
        activation=None,
        name=name+"_7",
    )(conv_6)

    flat = keras.layers.Flatten(
        name=name+"_conv_flat",
    )(conv_7)

    feature = flat
    if test_mode:
        print('obs shape:\t', img_in.get_shape())
        print('conv_1 shape:\t', conv_1.get_shape())
        print('conv_2 shape:\t', conv_2.get_shape())
        print('pool_1 shape:\t', pool_1.get_shape())
        print('conv_3 shape:\t', conv_3.get_shape())
        print('conv_4 shape:\t', conv_4.get_shape())
        print('conv_5 shape:\t', conv_5.get_shape())
        print('conv_6 shape:\t', conv_6.get_shape())
        print('dense shape:\t', flat.get_shape())
        print('feature shape:\t', feature.get_shape())
    return feature



if __name__ == "__main__":
    session = tf.compat.v1.Session()
    obs_in = keras.layers.Input(
        shape=[None, 16, 90, 3],
    )
    encoder = make_encoder_net(obs_in, test_mode=True, name="egg")
    pass

