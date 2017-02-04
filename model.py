
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, ELU, MaxPooling2D
from keras.layers import Convolution2D, Lambda
from keras.optimizers import Adam

import matplotlib.image as mpimg
import numpy as np
import pandas
import os
import json
import cv2

import matplotlib
import scipy.misc

import matplotlib.pyplot as plt

image_height = 66
image_width = 200

def show_image(img):
    plt.imshow(img)
    plt.show()


def load_image(imagepath, data_path):
    imagepath = imagepath.replace(' ', '')
    return mpimg.imread(data_path + imagepath)

def random_brightness():
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = .1 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image = cv2.resize(image, (img_height, img_width), interpolation=cv2.INTER_AREA)
    return image


# crop the dashboard and sky to focus image on road
def process_image(image, width, height):
    cropped_image = image[32:140, ]
    return scipy.misc.imresize(cropped_image, [width, height])


# select left, right and center images at random, adjusting steering if left and right
def gen_training_data(line_data, data_path, width, height):
    random_image = np.random.randint(3)
    steer_adjust = 0.0
    if (random_image == 0):
        image_path = line_data['center'][0].strip()
    if (random_image == 1):
        image_path = line_data['left'][0].strip()
        steer_adjust = 0.3
    if (random_image == 2):
        image_path = line_data['right'][0].strip()
        steer_adjust = -0.3
    steering = line_data['steering'][0] + steer_adjust
    image = load_image(image_path, data_path)
    # crop and resize image
    image = process_image(image, width, height)
    image = np.array(image)
    # randomly flip image
    flip_it = np.random.randint(2)
    if flip_it == 1:
        image = cv2.flip(image, 1)
        steering = -steering

    return image, steering


# bias selection of images so images with larger steering value are preferred
def biased_images(line_data, data_path, width, height, threshold, probability):
    image, steering = gen_training_data(line_data, data_path, width, height)
    prob = np.random.uniform()
    while (True):
        if (abs(steering) > threshold or prob > probability):
            return image, steering
        else:
            image, steering = gen_training_data(line_data, data_path, width, height)
            prob = np.random.uniform()


# generating batches of images with steering adjustments
def gen_batch(data, data_path, width, height, batch_size=32):
    batch_images = np.zeros((batch_size, width, height, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for current in range(batch_size):
            line_index = np.random.randint(len(data))
            line_data = data.iloc[[line_index]].reset_index()
            image, steering = biased_images(line_data, data_path, width, height, 0.1, 0.8)
            batch_images[current] = image
            batch_steering[current] = steering
        yield batch_images, batch_steering

# CNN architecture following Nvidia's model
def create_model(time_len=1):
    model = Sequential()
    # normalize input to (-1, 1)
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(image_width, image_height, 3),
        output_shape=(image_width, image_height, 3)))
    # 3 layers of 5 x 5 kernel, 2 x 2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(LeakyReLU())
    
    model.add(Dropout(.5))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(LeakyReLU())
    
    # final 2 layers, non-strided, 3 x 3 kernel
    model.add(Convolution2D(64, 3, 3))
    model.add(LeakyReLU())
    model.add(Dropout(.5))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(LeakyReLU())
    model.add(Dropout(.5))

    # fully-connected layers
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(LeakyReLU())
    model.add(Dense(512))
    model.add(LeakyReLU())
    # dropout to avoid overfitting
    model.add(Dropout(.5))
    model.add(LeakyReLU())
    model.add(Dense(1))

    return model



data_path = "data/"
data = pandas.read_csv(data_path + "/driving_log.csv")

def split_data(data):
    validationIndexes = int(data.shape[0] / 10)
    #shuffle the dataframe
    shuffled_data = data.reindex(np.random.permutation(data.index))
    #return training and validation data
    return shuffled_data[validationIndexes:], shuffled_data[:validationIndexes]

training_data, validation_data = split_data(data)

model = create_model()
model.summary()
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)
nb_epoch = 2

history = model.fit_generator(gen_batch(training_data, data_path, image_width, image_height), samples_per_epoch=300*32,
                              validation_data=gen_batch(validation_data, data_path, image_width, image_height), nb_val_samples=30*32,
                              nb_epoch=nb_epoch, verbose=1)

model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
    print("Model saved")