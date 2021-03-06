#!/usr/bin/env python
# coding: utf-8

# In[1]:


# example of defining a u-net encoder-decoder generator model
import os
import random
from random import randint
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
plt.style.use("grayscale")
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib notebook
import time

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from skimage import color

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.losses import mean_absolute_error
import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import pickle

# # Load Images

# In[2]:


def get_data(path, im_height, im_width, to_lab = False):
    ids_x = next(os.walk(path + "TRM"))[2]
    ids_x.sort()
    ids_y = next(os.walk(path + "RGB"))[2]
    ids_y.sort()
    X = np.zeros((len(ids_x), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids_y), im_height, im_width, 3), dtype=np.float32)
    
    for n, id_x in tqdm_notebook(enumerate(ids_x), total=len(ids_x)):
        id_y = ids_y[n]
        img = cv.imread(path + '/TRM/' + id_x,cv.COLOR_BGR2GRAY)
        x_img = np.array(img)
        x_img = (x_img - x_img.min()) / (x_img.max() - x_img.min())
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        y_img = img_to_array(load_img(path + '/RGB/' + id_y, color_mode="rgb"))
        y_img = resize(y_img, (im_height, im_width, 3), mode='constant', preserve_range=True)
        if to_lab:
            y_img = color.rgb2lab(y_img)
        else:
            y[n] = y_img / 255
        
        X[n, ..., 0] = x_img.squeeze()
        
    return X, y


# # Loss Func

# In[3]:
def vgg_loss_inner(y_true, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(256,256,3))
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False
    
    return 10*K.mean(K.square(model(y_true) - model(y_pred)))

def vgg_obj(vgg_fact=1):
    def vgg_loss(y_true, y_pred):
        return K.abs(vgg_fact * vgg_loss_inner(y_true, y_pred) + (1-vgg_fact) * mean_absolute_error(y_true, y_pred))
    return vgg_loss

def ssim_metric(y_true, y_pred):
    batch_size = y_true.shape[0];
    ssim_sum = 0
    for i in range(batch_size):
        ssim_sum += tf.image.ssim_multiscale(y_true,y_pred,1)
    return ssim_sum/batch_size;


def ssim_tf(ssim_fact=1):
    def ssim_loss(y_true, y_pred):
        return K.abs(ssim_fact * (1-tf.image.ssim_multiscale(y_true,y_pred,1)) + (1-ssim_fact) * mean_absolute_error(y_true, y_pred))
    return ssim_loss

# # Discriminator

# In[4]:


def define_discriminator(t_image_shape, c_image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=t_image_shape)
	# target image input
	in_target_image = Input(shape=c_image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model


# # Generator blocks

# In[5]:


# Encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
# Decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g


# # Generator model

# In[6]:


def define_generator(image_shape=(256,256,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model


# # GAN model

# In[7]:
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape, loss_fact=0.5, loss_weights=[1,100]):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # ssim loss
#     ssimTF = ssim_tf(ssim_fact=loss_fact)
    vggloss = vgg_obj(vgg_fact=loss_fact);
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', vggloss], optimizer=opt, loss_weights=loss_weights)
    return model


# In[8]:


def generate_real_samples(train_crops, n_samples, patch_shape):
    # retrieve selected images
    x, y = next(train_crops)
    ix = np.random.randint(0, x.shape[0], size=n_samples)
    
    if len(x.shape) != 4:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    z = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [x, y], z

def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	z = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, z


# train pix2pix models
def train(d_model, g_model, gan_model, train_crops, n_steps, n_batch, weight_name, dir_name=None):
    if dir_name is None:
        dir_name = weight_name
    # number of iter to save state
    save_num = 200
    # size of descriminator return - 16x16
    n_patch=16
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # init tmp vars
    g_loss_min = 100;
    d1 = []; d2 = []; g1 = []; g2 = []; g3 = [];
    ds1 = 0; ds2 = 0; gs1 = 0; gs2 = 0; gs3 = 0;
    
    # manually enumerate epochs
    for i in range(n_steps):
        start_time = time.time()
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(train_crops, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_total, g_cross, g_loss = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # save tmp learning losses
        ds1 += d_loss1; ds2 += d_loss2; gs1 += g_total; gs2 += g_cross; gs3 += g_loss;
        print('%d, d_real[%.3f] d_fake[%.3f] g_total[%.3f] g_cross[%.3f] g_loss[%.3f] %.3f[sec]' % (i, d_loss1, d_loss2, g_total, g_cross, g_loss, time.time()-start_time))
        
        # save best params
        if i % 10 == 0:
            d1.append(ds1/10);d2.append(ds2/10);g1.append(gs1/10);g2.append(gs2/10);g3.append(gs3/10);
            ds1 = 0; ds2 = 0; gs1 = 0; gs2 = 0; gs3 = 0;
            if g_loss < g_loss_min:
                g_model.save_weights(dir_name+'/g_model_best_loss_' + weight_name + '.h5')
                g_loss_min = g_loss
                print('best loss model saved!')
        # save elaps params        
        if i % save_num == 0:
            g_model.save_weights(dir_name+'/g_model_' + weight_name + '.h5')
            d_model.save_weights(dir_name+'/d_model_' + weight_name + '.h5')
            print('model saved...')
    
    g_model.save_weights(dir_name+'/g_model_' + weight_name + '.h5')
    d_model.save_weights(dir_name+'/d_model_' + weight_name + '.h5')
    return d1, d2, g1, g2, g3

# In[9]:

def create_augment(x, y, dict_augments, batch_size=1, shuffle=True, rand_crop=True, seed=1234):
    crop_size=256
    image_datagen = ImageDataGenerator(**dict_augments)        
    yimage_generator = image_datagen.flow(y, seed=seed, batch_size=batch_size, shuffle=shuffle)
    Ximage_generator = image_datagen.flow(x, seed=seed, batch_size=batch_size, shuffle=shuffle)
    generator = zip(Ximage_generator, yimage_generator)
    aug_crops = crop_generator(generator, crop_size, rand_crop=rand_crop)
    return aug_crops


def random_crop(it,ic, random_crop_size, rand_crop=True):
    # Note: image_data_format is 'channel_last'
    height, width = it.shape[0], it.shape[1]
    dy, dx = random_crop_size
    if rand_crop:
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
    else:
        x = (width - dx)//2
        y = (height - dx)//2
    crop_it = it[y:(y+dy), x:(x+dx), :]
    crop_ic = ic[y:(y+dy), x:(x+dx), :]
    return crop_it, crop_ic


def crop_generator(batches, crop_length, rand_crop=True):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crop_x = np.zeros((batch_x.shape[0], crop_length, crop_length, 1))
        batch_crop_y = np.zeros((batch_y.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crop_x[i], batch_crop_y[i] = random_crop(batch_x[i], batch_y[i], (crop_length, crop_length), rand_crop=rand_crop)
        yield (batch_crop_x, batch_crop_y)


# In[10]:

def save(data, filename):
    outfile = open(filename,'wb')
    pickle.dump(data,outfile)
    outfile.close()
def load(filename):
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict

def plot_sample(train_crops, g_model, to_lab=False):
    x, y = next(train_crops)
    xi = g_model.predict(x)
    yim = y
    if to_lab:
            yim = color.lab2rgb(yim)
            xi = color.lab2rgb(xi.squeeze())
    for i in range(x.shape[0]):
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        ax[0].imshow(x[i].squeeze())
        ax[0].set_title('IT')

        ax[2].imshow(yim[i])
        ax[2].set_title('Color')

        ax[1].imshow(xi[i])
        ax[1].set_title('Pred')
    return xi



def plot_loss(vec):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax[0,0].plot(vec[0])
    ax[0,0].set_title('d_real')

    ax[0,1].plot(vec[1])
    ax[0,1].set_title('d_fake')

    ax[0,2].plot(vec[2])
    ax[0,2].set_title('g_total')

    ax[1,0].plot(vec[3])
    ax[1,0].set_title('g_cross')

    ax[1,1].plot(vec[4])
    ax[1,1].set_title('g_loss')  

def plot_compare(g_names, dir_name, g_model, train_crops):
    x, y = next(train_crops)
    for i in range(len(g_names)):
        g_model.load_weights(dir_name+'/g_model_' + g_names[i] + '.h5')
        xi = g_model.predict(x)
        
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        yim = y
        ax[0].imshow(x[0].squeeze())
        ax[0].set_title(g_names[i])

        ax[2].imshow(yim[0])
        ax[2].set_title('Color')

        ax[1].imshow(xi[0])
        ax[1].set_title('Pred')

# In[15]:


def plot_schedual(train_crops, dir_name, model_name_list, to_lab=False):
    for i in range(model_name_list):
        t_image_shape = (256,256,1);c_image_shape = (256,256,3)
        g_model = define_generator(t_image_shape)
        g_model.load_weights(dir_name+'/g_model_' + model_name_list[i] + '.h5')
        plot_sample(train_crops, g_model)
