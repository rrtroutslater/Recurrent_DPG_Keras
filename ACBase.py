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
        load network weights from file        self.tau = tau

            net_t_fname: filename of saved target model
        """
        return

    def save_model(self, 
            net_fname="",
            net_t_fname=""
            ):
        """
        save the network and target network to file, return filenames of saved model
        """
        if net_fname == "":
            net_fname = self.net_type + "_" + datetime.datetime.now()
        if net_t_fname == "":
            net_fname = self.net_type + "_target_" + datetime.datetime.now()

        tf.keras.models.save_model(self.net, net_fname)
        tf.keras.models.save_model(self.net_t, net_t_fname)
        return net_fname, net_t_fname

    def update_target_net(self,
            tau=0.3
        ):
        """
        copy weights of actor network into target network

        inputs:
            tau: in [0,1], weight of network weights in target network weight update
        """
        weights = self.net.get_weights()
        target_weights = self.net_t.get_weights()
        for i in range(0, len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]

        self.net_t.set_weights(target_weights)
        return
