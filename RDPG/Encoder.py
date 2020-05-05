from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import datetime
import shutil


class Encoder():
    def __init__(self,
                 session,
                 img_dim=[16, 90, 3],
                 obs_dim=32,
                 learning_rate=0.005,
                 beta=1.0,
                 test_mode=False,
                 ):
        tf.compat.v1.disable_eager_execution()

        self.sess = session
        self.img_dim = img_dim
        self.obs_dim = obs_dim
        self.learning_rate = learning_rate
        self.test_mode = test_mode

        self.feature, self.img_in = self.make_encoder_net()
        self.reconstructed_img = self.make_decoder_net(self.feature)

        self.encoder_model = keras.Model(self.img_in, self.feature)

        self.encoder_decoder = keras.Model(
            inputs=self.img_in, 
            outputs=self.reconstructed_img
        )

        self.encoder_decoder.compile(
            optimizer='adam',
            loss='mean_squared_error',
        )
        return

    def get_obs(self, img_in):
        """
        returs flattened features extracted from encoder
        """
        print(img_in.shape)

        if len(img_in.shape) > 4:
            img = np.reshape(
                img_in,
                [img_in.shape[0] * img_in.shape[1], img_in.shape[2], img_in.shape[3], img_in.shape[4]]
            )
        else:
            img = img_in

        obs_3d = self.encoder_model.predict(img)
        obs = np.reshape(
            obs_3d, 
            [obs_3d.shape[0], obs_3d.shape[1] * obs_3d.shape[2] * obs_3d.shape[3]]
        )

        if len(img_in.shape) > 4:
            obs = np.reshape(
                obs,
                [img_in.shape[0], img_in.shape[1], obs.shape[1]]
            )

        if 0:
            fig, ax = plt.subplots(obs_3d[0].shape[2],1)
            for k in range(len(ax)):
                ax[k].imshow(obs_3d[0][:,:,k], cmap='gray')
                ax[k].axis('off')
            plt.show()

            img_out = self.encoder_decoder.predict(img_in[0])[0]
            print(img_out.shape)

            fig, ax = plt.subplots(3,2)
            ax[0,0].imshow(img_in[0][0][:,:,0], cmap='gray')
            ax[1,0].imshow(img_in[0][0][:,:,1], cmap='gray')
            ax[2,0].imshow(img_in[0][0][:,:,2], cmap='gray')
            ax[0,1].imshow(img_out[:,:,0], cmap='gray')
            ax[1,1].imshow(img_out[:,:,1], cmap='gray')
            ax[2,1].imshow(img_out[:,:,2], cmap='gray')
            plt.show()
        return obs

    def make_encoder_net(self):
        """
        encoder network
        [batch_size,16,90,3] -> [batch_size,64]
        """
        img_in = keras.layers.Input(shape=self.img_dim)

        conv_1 = keras.layers.Conv2D(
            filters=8,
            kernel_size=[5,5],
            strides=[1,1],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(img_in)
        conv_2 = keras.layers.Conv2D(
            filters=16, 
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_1)
        conv_3 = keras.layers.Conv2D(
            filters=24,
            kernel_size=[3,5],
            strides=[2,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_2)
        conv_4 = keras.layers.Conv2D(
            filters=32, 
            kernel_size=[3,3],
            strides=[2,2],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3)
        )(conv_3)
        conv_5 = keras.layers.Conv2D(
            filters=8,
            kernel_size=[3,3],
            strides=[1,1],
            padding='same',
            activation=keras.layers.LeakyReLU(alpha=0.3),
        )(conv_4)

        feature = conv_5

        if self.test_mode:
            print('obs shape:\t', img_in.get_shape())
            print('conv_1 shape:\t', conv_1.get_shape())
            print('conv_2 shape:\t', conv_2.get_shape())
            print('conv_3 shape:\t', conv_3.get_shape())
            print('conv_4 shape:\t', conv_4.get_shape())
            print('conv_5 shape:\t', conv_5.get_shape())
            print('feature shape:\t', feature.get_shape())

        return feature, img_in

    def make_decoder_net(self, feature):
        upconv_0 = keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=[1,1],
            strides=[1,1],
            padding='same',
            activation=tf.nn.relu,
        )(feature)
        upconv_1 = keras.layers.Conv2DTranspose(
            filters=24,
            kernel_size=[2,3],
            strides=[2,2],
            padding='same',
            activation=tf.nn.relu,
        )(upconv_0)
        upconv_2 = keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=[3,3],
            strides=[2,2],
            padding='same',
            activation=tf.nn.relu,
        )(upconv_1)
        upconv_3 = keras.layers.Conv2DTranspose(
            filters=8,
            kernel_size=[3,5],
            strides=[2,2],
            padding='same',
            activation=tf.nn.relu,
        )(upconv_2)
        upconv_4 = keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=[3,3],
            strides=[1,1],
            padding='same',
            activation=tf.nn.relu,
        )(upconv_3)

        output = upconv_4[:,:,:90,:]

        if self.test_mode:
            print('feature shape:\t', feature.get_shape())
            print('upconv_0_shape:\t', upconv_0.get_shape())
            print('upconv_1_shape:\t', upconv_1.get_shape())
            print('upconv_2_shape:\t', upconv_2.get_shape())
            print('upconv_3_shape:\t', upconv_3.get_shape())
            print('upconv_4_shape:\t', upconv_4.get_shape())
            print('output shape:\t', output.get_shape())

        return output
    
    def train_autoencoder(self,
            dataset,
            dataset_validation,
            batch_size=100,
            num_batch_sample=10,
            steps_per_batch=10
            ):
        """
        TODO
        """
        loss = []
        loss_validation = []

        for i in range(0, num_batch_sample):
            img_idx = np.random.choice(dataset.shape[0], size=batch_size, replace=False)
            img_in = dataset[img_idx]

            val_idx = np.random.choice(dataset_validation.shape[0], size=batch_size, replace=False)
            img_val = dataset_validation[val_idx]

            for j in range(0, steps_per_batch):
                loss.append(self.encoder_decoder.train_on_batch(x=img_in, y=img_in))
                l_val = self.encoder_decoder.evaluate(
                        x=img_val, 
                        y=img_val, 
                        batch_size=batch_size,
                        verbose=0,
                    )
                loss_validation.append(l_val)

        if self.test_mode:
            img_original = np.array([dataset_validation[0]])
            img_reconstructed = self.encoder_decoder.predict(img_original)

            fig, ax = plt.subplots(3,2)
            ax[0, 0].imshow(img_original[0,:,:,0], cmap='gray')
            ax[0, 0].axis('off')
            ax[1, 0].imshow(img_original[0,:,:,1], cmap='gray')
            ax[1, 0].axis('off')
            ax[2, 0].imshow(img_original[0,:,:,2], cmap='gray')
            ax[2, 0].axis('off')
            ax[0, 1].imshow(img_reconstructed[0,:,:,0], cmap='gray')
            ax[0, 1].axis('off')
            ax[1, 1].imshow(img_reconstructed[0,:,:,1], cmap='gray')
            ax[1, 1].axis('off')
            ax[2, 1].imshow(img_reconstructed[0,:,:,2], cmap='gray')
            ax[2, 1].axis('off')
            plt.show()

            feature = self.encoder_model.predict(img_original)[0]

            fig, ax = plt.subplots(feature.shape[2],1)
            for k in range(len(ax)):
                ax[k].imshow(feature[:,:,k], cmap='gray')
                ax[k].axis('off')
            plt.show()
            self.test_mode = False

        print('=========================')
        print('\nloss:\t', loss[-1])
        print('val loss:\t', loss_validation[-1])
        return loss, loss_validation

    def save_model(self, model_dir='./trained_encoders/'):
        encoder_fn = model_dir + "encoder_" + str(datetime.datetime.now())
        encoder_decoder_fn = model_dir + "encoder_DECODER" + str(datetime.datetime.now())

        print('\nWEIGHTS BEFORE SAVING:\n', self.encoder_model.get_weights()[0][0][0])
        self.encoder_model.save_weights(encoder_fn)
        self.encoder_decoder.save_weights(encoder_decoder_fn)
        return encoder_fn, encoder_decoder_fn

    def load_model(self, encoder_fn, encoder_decoder_fn):
        print('\nWEIGHTS BEFORE RESTORE:\n', self.encoder_model.get_weights()[0][0][0])
        self.encoder_model.load_weights(encoder_fn)
        self.encoder_decoder.load_weights(encoder_decoder_fn)
        print('\nWEIGHTS AFTER RESTORE:\n', self.encoder_model.get_weights()[0][0][0])
        return

    def get_weight_check(self,):
        print('SAMPLE ENCODER WEIGHTS:\n', self.encoder_model.get_weights()[0][0][0])
        return


def train():
    session = tf.compat.v1.Session()
    encoder = Encoder(
        session, 
        test_mode=True, 
        learning_rate=0.000005
    )

    RL_DATA_DIR = './training_data/rlbp_data/'
    LBP_DATA_DIR = './training_data/lbp_data/'
    RL_VAL_DIR = './validation_data/rlbp_data/'
    LBP_VAL_DIR = './validation_data/lbp_data/'

    RL_FN_LIST = os.listdir(RL_DATA_DIR)
    LBP_FN_LIST = os.listdir(LBP_DATA_DIR)
    RL_VAL_LIST = os.listdir(RL_VAL_DIR)
    LBP_VAL_LIST = os.listdir(LBP_VAL_DIR)

    NUM_EPOCH = 452
    NUM_DATASET_PER_BATCH = 20

    loss = []
    loss_val = []
    for i in range(0, NUM_EPOCH):

        # training data
        training_fn_list = np.random.choice(RL_FN_LIST, NUM_DATASET_PER_BATCH, replace=False)
        training_fn_list = [RL_DATA_DIR + fn for fn in training_fn_list]
        training_fn_list.append(LBP_DATA_DIR + np.random.choice(LBP_FN_LIST, 1, replace=False)[0])

        # validation data
        validation_fn_list = np.random.choice(RL_VAL_LIST, NUM_DATASET_PER_BATCH, replace=False)
        validation_fn_list = [RL_VAL_DIR + fn for fn in validation_fn_list]
        validation_fn_list.append(LBP_VAL_DIR + np.random.choice(LBP_VAL_LIST, 1, replace=False)[0])

        # validation data
        dataset_validation = None
        for fn in validation_fn_list:
            d = None
            with open(fn, 'rb') as f:
                d = pickle.load(f)
                if dataset_validation is None:
                    if 'x_val' in d.keys():
                        dataset_validation = d['x_val']
                    else:
                        dataset_validation = d['img_0']
                else:
                    if 'x_val' in d.keys():
                        dataset_validation = np.concatenate((dataset_validation, d['x_val']), axis=0)
                    elif 'img_0' in d.keys():
                        dataset_validation = np.concatenate((dataset_validation, d['img_0']), axis=0)
                    else:
                        print('NO VALID KEY FOUND IN:\n', d.keys())

        # training data 
        dataset = None
        for fn in training_fn_list:
            d = None
            with open(fn, 'rb') as f:
                d = pickle.load(f)
                if dataset is None:
                    if 'x_train' in d.keys():
                        dataset = d['x_train']
                    else:
                        dataset = d['img_0']
                else:
                    if 'x_train' in d.keys():
                        # print(d['x_train'].shape)
                        dataset = np.concatenate((dataset, d['x_train']), axis=0)
                    else:
                        # print(d['img_0'].shape)
                        dataset = np.concatenate((dataset, d['img_0']), axis=0)

        l, l_val = encoder.train_autoencoder(dataset, dataset_validation)

        for j in range(0, len(l)):
            loss.append(l[j])
        for j in range(0, len(l_val)):
            loss_val.append(l_val[j])

        if i > 0 and i % 150 == 0:
            encoder_fn, encoder_decoder_fn = encoder.save_model()
            # encoder.test_mode = True

        if i > 0 and i % 150 == 0:
            encoder.test_mode = True

        if i > 0 and i % 15 == 0:
            if i > 150:
                encoder.learning_rate *= 0.95
            else:
                encoder.learning_rate *= 0.90
            print('learning rate updated:\t', encoder.learning_rate)

    plt.plot(loss[100:], label='training')
    plt.plot(loss_val[100:], label='validation')
    plt.ylim(0, 0.02)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    # train()

    session = tf.compat.v1.Session()
    encoder = Encoder(
        session, 
        test_mode=False, 
        learning_rate=0.000005
    )

    # testing forward pass:
    if 0:
        img = np.random.randn(1, 4, 16, 90, 3)
        encoder.get_obs(img)

    # test model loading
    if 1:
        encoder.load_model(
            './trained_encoders/'+'encoder_2020-05-04 18:40:59.673265',
            './trained_encoders/'+'encoder_DECODER2020-05-04 18:40:59.673290',
        )


    pass













