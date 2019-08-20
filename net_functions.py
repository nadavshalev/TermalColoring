import keras
import os
import numpy as np
import skimage
from save_func import *
import time


def load_data(path_test, path_train, val_size, train_num=-1, to_save=True, save_path='./imFile'):
    im_process = keras.preprocessing.image

    # load train set
    X = []
    for (i, im) in enumerate(os.listdir(path_train)):
        if train_num != -1 and train_num <= i:
            break
        if i % 100 == 0:
            print('     load: ', i)
        X.append(im_process.img_to_array(im_process.load_img(path_train + im)))

    X = np.array(X, dtype=float)

    # split train to validation
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    n_samples_train = int(n_samples * (1-val_size))
    n_samples_validation = int(n_samples * val_size)
    train_indices = indices[:n_samples_train]
    validation_indices = indices[n_samples_train:]
    train_set = 1.0/255*X[train_indices,:,:,:]
    validation_set = 1.0/255*X[validation_indices,:,:,:]

    # load test set
    test_set = []
    for (i, im) in enumerate(os.listdir(path_test)):
        test_set.append(im_process.img_to_array(im_process.load_img(path_test + im)))

    test_set = np.array(test_set, dtype=float)
    test_set = skimage.color.rgb2lab(1.0/255*test_set)[:,:,:,0]
    test_set = test_set.reshape(test_set.shape+(1,))
    skimage.io.imsave
    if to_save:
        data = {
            "train_set": train_set,
            "validation_set": validation_set,
            "test_set": test_set
        }
        save_v(save_path, data)

    return train_set,validation_set,test_set

def load_from_file(save_path='./imFile'):
    print('loading images...')
    start_time = time.time()
    data = open_v(save_path)
    print('loaded (', time.time() - start_time, '[sec] )')
    return data["train_set"],data["validation_set"],data["test_set"]

def get_data_labels(data):
    lab_data = skimage.color.rgb2lab(data)
    X = lab_data[:, :, :, 0]  # gray chann
    X = X.reshape(X.shape + (1,))  # add 1 dim
    Y = lab_data[:, :, :, 1:]/128  # a and b chann
    return X,Y


def get_model():
    ly = keras.layers
    md = keras.models

    # Encoder
    encoder_input = ly.Input(shape=(256, 256, 1,))
    encoder_output = ly.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = ly.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = ly.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = ly.Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = ly.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = ly.Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = ly.Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = ly.Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)

    # Decoder
    decoder_output = ly.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
    decoder_output = ly.UpSampling2D((2, 2))(decoder_output)
    decoder_output = ly.Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = ly.UpSampling2D((2, 2))(decoder_output)
    decoder_output = ly.Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = ly.Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = ly.Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = ly.UpSampling2D((2, 2))(decoder_output)

    return md.Model(inputs=encoder_input, outputs=decoder_output)


def train(x_train, y_train, model, datagen, batch_size, epochs, ev_time, x_val, y_val):
    for e in range(epochs):
        print('Epoch', e+1)
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
            model.fit(x_batch, y_batch,verbose=0)
            batches += 1
            if batches >= len(x_train) / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        if e % ev_time == 0:
            score = model.evaluate(x_val, y_val, verbose=0)
            print('Validation: ', score)
    return model


def save_results(results, inputs, dir_path):
    for i in range(len(results)):
        cur = np.zeros((256, 256, 3))
        cur[:, :, 0] = inputs[i][:, :, 0]
        cur[:, :, 1:] = results[i]
        skimage.io.imsave(dir_path + "im_" + str(i) + ".png", skimage.color.lab2rgb(cur))