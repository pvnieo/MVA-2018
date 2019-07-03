

import pdb
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras import backend as K


class autoencoder():
    def __init__(self, dataset_name='mnist', architecture='mlp'):

        X_train = self.load_data(dataset_name)
        optimizer = 'adadelta'  # Adam(0.0002, 0.5) #

        # image parameters
        self.epochs = 4000
        self.error_list = np.zeros((self.epochs, 1))
        self.img_rows = X_train.shape[1]
        self.img_cols = X_train.shape[2]
        self.img_channels = X_train.shape[3]
        self.img_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.z_dim = 10
        self.architecture = architecture
        self.dataset_name = dataset_name
        self.sigma = 40.0/255.0

        # Build and compile the autoencoder
        self.ae = self.build_ae()
        self.ae.summary()
        # binary cross-entropy loss, because mnist is grey-scale
        # you can try out the mse loss as well if you like
        self.ae.compile(optimizer=optimizer, loss='binary_crossentropy')

    def build_ae(self):

        n_pixels = self.img_rows*self.img_cols*self.img_channels

        if (self.architecture == 'mlp'):
            # FULLY CONNECTED (MLP)

            # BEGIN INSERT CODE
            # encoder
            input_img = Input(shape=(self.img_rows, self.img_cols, self.img_channels))
            input_img_flatten = Flatten()(input_img)
            z = Dense(self.z_dim)(input_img_flatten)
            z = LeakyReLU(alpha=0.2)(z)
            # decoder
            output = Dense(784)(z)
            output = Activation('sigmoid')(output)
            output_img = Reshape((28, 28, 1))(output)

            # END INSERT CODE
        elif(self.architecture == 'convolutional'):
            # CONVOLUTIONAL MODEL

            # BEGIN INSERT CODE
            input_img = Input(shape=(self.img_rows, self.img_cols, self.img_channels))
            # encoder
            z = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="same", bias=True)(input_img)
            z = LeakyReLU(alpha=0.2)(z)
            z = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding="same", bias=True)(input_img)
            z = LeakyReLU(alpha=0.2)(z)
            z = Flatten()(z)
            z = Dense(self.z_dim)(z)
            # decoder
            output = Dense(196)(z)
            output = LeakyReLU(alpha=0.2)(output)
            output = Reshape((7, 7, 4))(output)
            output = Conv2DTranspose(kernel_size=(3, 3), filters=8, strides=(2, 2), padding="same")(output)
            output = LeakyReLU(alpha=0.2)(output)
            output = Conv2DTranspose(kernel_size=(3, 3), filters=1, strides=(2, 2), padding="same")(output)
            output_img = Activation('sigmoid')(output)
            # decoder
            # END INSERTs CODE
        elif(self.architecture == 'denoising'):
            input_img = Input(shape=(self.img_rows, self.img_cols, self.img_channels))
            noisy_input = GaussianNoise(self.sigma)(input_img)
            # encoder
            z = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="same", bias=True)(noisy_input)
            z = LeakyReLU(alpha=0.2)(z)
            z = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding="same", bias=True)(input_img)
            z = LeakyReLU(alpha=0.2)(z)
            z = Flatten()(z)
            z = Dense(self.z_dim)(z)
            # decoder
            output = Dense(196)(z)
            output = LeakyReLU(alpha=0.2)(output)
            output = Reshape((7, 7, 4))(output)
            output = Conv2DTranspose(kernel_size=(3, 3), filters=8, strides=(2, 2), padding="same")(output)
            output = LeakyReLU(alpha=0.2)(output)
            output = Conv2DTranspose(kernel_size=(3, 3), filters=1, strides=(2, 2), padding="same")(output)
            output_img = Activation('sigmoid')(output)
            # decoder

        # output the model
        return Model(input_img, output_img)

    def load_data(self, dataset_name):
        # Load the dataset
        if(dataset_name == 'mnist'):
            (X_train, _), (_, _) = mnist.load_data()
        else:
            print('Error, unknown database')

        # normalise images between 0 and 1
        X_train = X_train/255.0
        # add a channel dimension, if need be (for mnist data)
        if(X_train.ndim == 3):
            X_train = np.expand_dims(X_train, axis=3)
        return X_train

    def train(self, epochs, batch_size=128, sample_interval=50):

        # load dataset
        X_train = self.load_data(self.dataset_name)


        for i in range(0, epochs):

            # ---------------------
            #  Autoencoder
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            curr_batch = X_train[idx, :, :, :]
            # Autoencoder training
            loss = self.ae.train_on_batch(curr_batch, curr_batch)

            # print the losses
            print("%d [Loss: %f]" % (i, loss))
            self.error_list[i] = loss

            # Save some random generated images and the models at every sample_interval iterations
            if (i % sample_interval == 0):
                n_images = 5
                idx = np.random.randint(0, X_train.shape[0], n_images)
                test_imgs = X_train[idx, :, :, :]
                curr_batch = test_imgs
                self.test_images(
                    curr_batch, 'images/'+self.dataset_name+'_reconstruction_%06d.png' % i)

    def test_images(self, test_imgs, image_filename):
        n_images = test_imgs.shape[0]
        # get output imagesq
        output_imgs = self.ae.predict(test_imgs)

        r = 2
        c = n_images
        fig, axs = plt.subplots(r, c)
        for j in range(c):
            # black and white images
            axs[0, j].imshow(test_imgs[j, :, :, 0], cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].imshow(output_imgs[j, :, :, 0], cmap='gray')
            axs[1, j].axis('off')
        fig.savefig(image_filename)
        plt.close()


if __name__ == '__main__':

    # create the output image directory
    if (os.path.isdir('images') == 0):
        os.mkdir('images')

    # choose dataset
    dataset_name = 'mnist'

    # create AE model
    architecture = 'denoising' #'convolutional'  #   'mlp'
    ae = autoencoder(dataset_name, architecture)  # ,

    ae.train(epochs=ae.epochs, batch_size=64, sample_interval=100)
    plt.plot(ae.error_list[30:])
    plt.show()
