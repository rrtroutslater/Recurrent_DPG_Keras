from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime

class ACBase():
    def __init__(self,):
        """
        base class for actor critic models. contains common methods
        - saving / loading model
        - updating target network
        - printing network details (weights, grads, etc)
        - saving .png of network architecture
        """
        # set by child classes
        self.net_type = ""
        self.net = keras.Model()
        self.net_weights = []

        self.net_t = keras.Model()
        self.net_t_weights = []

        self.lstm_horizon = 0  

        self.grads = {}
        return

    def set_network_type(self,
            net_type
        ):
        assert net_type in ["actor", "critic"]
        self.net_type = net_type
        return

    def export_model_figure(self,
            fnames=[]
        ):
        """
        exports figure of model architecture 
        """
        assert len(fnames) in [0,2]
        if len(fnames) == 0:
            net_fname = "architecture_" + self.net_type + "_network.png"
            net_t_fname = "architecture_" + self.net_type + "_target_network.png"
        else: 
            net_fname = fnames[0]
            net_t_fname = fnames[1]
        keras.utils.plot_model(self.net, to_file=net_fname)
        keras.utils.plot_model(self.net_t, to_file=net_t_fname)
        return

    def print_network_info(self):
        """
        print model stats, layer shapes, etc
        """
        print("\nactor weights:")
        for w in self.net_weights:
            print(w)
            print(w.get_shape())

        print("\nactor target weights:")
        for w in self.net_t_weights:
            print(w)
            print(w.get_shape())

        print("\ngradients:")
        for k in self.grads.keys():
            print(k)
            if type(self.grads[k]) == list:
                print(self.grads[k])
            elif type(self.grads[k] == tf.Tensor):
                print(self.grads[k].get_shape())
            else:
                print('unsupported display type')
        return

    def load_model(self, 
            net_fname,
            net_t_fname,
        ):
        """
        load network weights from file

        inputs:
            net_t: filename of saved model
            net_t_fname: filename of saved target model
        """
        self.net.load_weights(net_fname)
        self.net_t.load_weights(net_t_fname)

        self.net_weights = self.net.trainable_weights
        self.net_t_weights = self.net_t.trainable_weights
        print('WEIGHTS AFTER RESTORE:')
        self.get_weight_check()
        return

    def save_model(self, 
            net_fname="",
            net_t_fname="",
            learning_rate="",
            tau="",
            lstm_horizon="",
            ):
        """
        save the network and target network to file, return filenames of saved model
        """
        print('WEIGHTS BEFORE SAVE:')
        self.get_weight_check()
        
        if net_fname == "":
            net_fname = "./trained_models/4/" + self.net_type + "_" + str(self.lstm_horizon) \
                + "_" + lstm_horizon +"final"#\
                # + str(datetime.datetime.now()) \
        if net_t_fname == "":
            net_t_fname = "./trained_models/4/" + self.net_type + "_target_" + str(self.lstm_horizon) \
                + "_" + learning_rate + "_" + tau + "_" \
                + "_" + lstm_horizon +"final"#\
                # + str(datetime.datetime.now()) \

        self.net.save_weights(net_fname, save_format='tf')
        self.net_t.save_weights(net_t_fname, save_format='tf')

        print('WEIGHTS AFTER SAVE:')
        self.get_weight_check()

        return net_fname, net_t_fname

    def get_weight_check(self,):
        print('EXAMPLE WEIGHTS:\n', self.net.get_weights()[0][0][0])
        return

    def update_target_net(self,
            tau=0.3,
            copy_all=False,
        ):
        """
        copy weights of actor network into target network

        inputs:
            tau: in [0,1], weight of network weights in target network weight update
        """
        weights = self.net.get_weights()
        target_weights = self.net_t.get_weights()
        for i in range(0, len(weights)):
            if copy_all:
                target_weights[i] = weights[i]
            else:
                target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]

        self.net_t.set_weights(target_weights)
        return

    def display_hidden_state(self,):
        print('current hidden state:\n', self.h_prev)
        print('current carry state:\n', self.c_prev)
        return

    def display_target_hidden_state(self,):
        print('current TARGET hidden state:\n', self.h_prev_t)
        print('current TARGET carry state:\n', self.c_prev_t)
        return

