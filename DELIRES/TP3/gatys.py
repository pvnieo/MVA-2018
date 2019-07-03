#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:09:50 2019

@author: said
"""

# Mohamed K. Eid (mohamedkeid@gmail.com)
# Description: TensorFlow implementation of "Texture-Synthesis Using Convolutional Neural Networks"

import argparse
import vgg19_TPMVA as vgg19

import numpy as np
import os
import tensorflow as tf
import time
from helpers_said import *
from functools import reduce
#%%
# Model hyperparams
TEXTURE_LAYERS = ['conv1_1', 'pool2', 'pool3', 'pool4', 'conv5_1']
EPOCHS = 2000
LEARNING_RATE = .02

NORM_TERM = 6.

# Loss term weights
TEXTURE_WEIGHT = 3.
NORM_WEIGHT = .1
SPECTRE_WEIGHT=0

# Default image paths
#DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = 'sortie_gatys.png'
INPUT_PATH, TEXTURE_PATH = None, None




# Calcul de la matrice de Gram d'un bloc convolutif
def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # on normalise par la taille spatiale pour obtenir une valeur comparable entre images 
    return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)/((dimension[1] * dimension[2]))


# Compute the L2-norm divided by squared number of dimensions
def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


# Calcul de la loss texture etant donne le resaeau x et les matrices de gram s
def get_texture_loss(x, s):
    with tf.name_scope('get_style_loss'):
        texture_layer_losses = [get_texture_loss_for_layer(x, s, l) for l in TEXTURE_LAYERS]
        texture_weights = tf.constant([1. / len(texture_layer_losses)] * len(texture_layer_losses), tf.float32)
        weighted_layer_losses = tf.multiply(texture_weights, tf.convert_to_tensor(texture_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)


# La loss texture pour une couche particuliere l
def get_texture_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):

        x_layer_maps = getattr(x, l)

        x_layer_gram = convert_to_gram(x_layer_maps)
        t_layer_gram = s[l] 

        shape = x_layer_maps.get_shape().as_list()
        size = shape[-1]**2 
        gram_loss = get_l2_norm_loss(x_layer_gram - t_layer_gram)
        return gram_loss / size

#%%
# LOSS du spectre NON UTILISEE
def get_spectre_loss(noise,texture,out_shape):
    fft2=np.fft.fft2
    ft=abs(fft2(texture,axes=(0,1)))
    print (ft.shape,out_shape[1:3])
    subsample= skimage.transform.resize(ft,out_shape[1:3],anti_aliasing=True)
    subsample=subsample/texture.shape[0]/texture.shape[1]*out_shape[1]*out_shape[2]
    subsample-subsample.reshape(out_shape)
    
    return tf.reduce_sum(tf.square(tf.abs(tf.fft2d(tf.cast(noise,tf.complex64)))-subsample))/\
          (out_shape[1]*out_shape[2])**2  
    
#%%
# Parse arguments and assign them to their respective global variables
def parse_args():
    global TEXTURE_PATH, OUT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("texture",
        help="path to the image you'd like to resample")
    parser.add_argument("--output",
        default=OUT_PATH,
        help="path to where the generated image will be created")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    TEXTURE_PATH = os.path.realpath(args.texture)
    OUT_PATH = os.path.realpath(args.output)


#%% Initialisation  des matrices de Gram

sess=tf.Session()
parse_args()
texture, image_shape = load_image(TEXTURE_PATH)
image_shape = [1] + image_shape
texture = texture.reshape(image_shape).astype(np.float32)
with tf.name_scope('vgg_texture'):
    texture_model = vgg19.Vgg19()
    texture_model.build(texture, image_shape[1:])

grams={}
# calcul des matrices de gram de l'image d'origine
for l in TEXTURE_LAYERS:

    tableau=sess.run(getattr(texture_model,l)) #gettattr renvoie texture_model.l

    tableau=tableau.reshape(tableau.shape[1:])
    shape=tableau.shape

    tableau=tableau.reshape((shape[0]*shape[1],-1))

    grams[l]=np.matmul (tableau.T,tableau)/ (shape[0]*shape[1])

    

sess.close()

#%%
with tf.Session() as sess:
    sample_size=[1,256,256,3]
    noise_init = tf.truncated_normal(sample_size, mean=.5, stddev=.1)
    noise = tf.Variable(noise_init, dtype=tf.float32)


    with tf.name_scope('vgg_x'):
        x_model = vgg19.Vgg19()
        x_model.build(noise, sample_size[1:])

    # Loss functions
    with tf.name_scope('loss'):
        # Texture
        if TEXTURE_WEIGHT is 0:
            texture_loss = tf.constant(0.)
        else:
            unweighted_texture_loss = get_texture_loss(x_model, grams)#texture_model)
            texture_loss = unweighted_texture_loss * TEXTURE_WEIGHT

        # Norm regularization
        if NORM_WEIGHT is 0:
            norm_loss = tf.constant(0.)
        else:
            norm_loss = (get_l2_norm_loss(noise) ** NORM_TERM) * NORM_WEIGHT
        if SPECTRE_WEIGHT is 0:
            spectre_loss = tf.constant(0.0)
        else:
            spectre_loss = SPECTRE_WEIGHT*get_spectre_loss(noise,texture[0],sample_size)
#        # Total variation denoising
#        if TV_WEIGHT is 0:
#            tv_loss = tf.constant(0.)
#        else:
#            tv_loss = get_total_variation(noise, image_shape) * TV_WEIGHT
#
        # Total loss
        total_loss = texture_loss + norm_loss  +spectre_loss#+ tv_loss

    # Update image
    with tf.name_scope('update_image'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grads = optimizer.compute_gradients(total_loss, [noise])
        clipped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        update_image = optimizer.apply_gradients(clipped_grads)

    # Train
   
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(EPOCHS):
        _, loss = sess.run([update_image, total_loss])
        print('etape numero',i,' parmi ',EPOCHS,'loss', loss)

    # FIN
    elapsed = time.time() - start_time
    print("Training complete. The session took %.2f seconds to complete." % elapsed)
    print("Rendering final image and closing TensorFlow session..")

    # Render the image after making sure the repo's dedicated output dir exists
    #out_dir = os.path.dirname(os.path.realpath(__file__)) + '/../output/'
    #if not os.path.isdir(out_dir):
    #    os.makedirs(out_dir)
    render_img(sess, noise, save=True, out_path=OUT_PATH)