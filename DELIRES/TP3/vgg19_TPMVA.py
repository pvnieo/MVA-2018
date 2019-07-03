#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:53:45 2019

@author: said
"""

import os
import tensorflow as tf
import numpy as np
import inspect
import urllib.request

VGG_MEAN = [103.939, 116.779, 123.68]
data = None
dir_path = os.path.dirname(os.path.realpath(__file__))
weights_name = os.path.abspath(dir_path + "/../lib/weights/vgg19.npy")
weights_url = "https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs"
#%% definition des repertoires
## MAC BOULOT
weights_file='/media/lawliet/Unlimited Burst/Study/DELIRES/TD/TP3_DELIRES/vgg19.npy'
## DSI TELECOM (POUR MVA)
#weights_file='/cal/homes/ladjal/TP_CNN/model/vgg19.npy'
## MON PC LINUX
#weights_file='/home/said/.local/lib/python3.5/site-packages/tensorflow_vgg/vgg19.npy'

class Vgg19:
    def __init__(self, vgg19_npy_path=weights_file):
        global data

        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, weights_name)
            path = '/media/lawliet/Unlimited Burst/Study/DELIRES/TD/TP3_DELIRES/vgg19.npy'

            if os.path.exists(path):
                vgg19_npy_path = path
            else:
                print("VGG19 weights were not found in the project directory!")
                print("Please download the .npy file from: %s" % weights_url)
                print("The file should be placed as '%s'" % weights_name)
                exit(0)

        if data is None:
            data = np.load(vgg19_npy_path, encoding='latin1')
            self.data_dict = data.item()
            print("VGG19 weights loaded")

        else:
            self.data_dict = data.item()

    def build(self, rgb, shape):
        rgb_scaled = rgb * 255.0
        num_channels = shape[2]
        channel_shape = shape
        channel_shape[2] = 1

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        assert red.get_shape().as_list()[1:] == channel_shape
        assert green.get_shape().as_list()[1:] == channel_shape
        assert blue.get_shape().as_list()[1:] == channel_shape

        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        shape[2] = num_channels
        assert bgr.get_shape().as_list()[1:] == shape

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1',self.filtre_moyenneur(64))

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2',self.filtre_moyenneur(128))

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.avg_pool(self.conv3_4, 'pool3',self.filtre_moyenneur(256))

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.avg_pool(self.conv4_4, 'pool4',self.filtre_moyenneur(512))

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")

        self.data_dict = None

    def filtre_moyenneur(self,nbf):
        filtre=np.zeros((2,2,nbf,nbf),np.float32)
        for k in range(nbf):
            filtre[:,:,k,k]=0.25
        return filtre

    def avg_pool(self, bottom, name,filtre_inutile):
        return tf.nn.avg_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
#    def avg_pool(self, bottom, name,filtre):
#        return tf.nn.conv2d(bottom,tf.constant(filtre),strides=[1,2,2,1],padding='SAME',name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
