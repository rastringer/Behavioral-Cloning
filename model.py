''' 
With huge thanks to Udacity Instructor David Silver 
for his excellent video Q & A on this project here http://bit.ly/2kwk5kz.
Some of the code below is from the tutorial.
'''

import csv
import cv2
import numpy as np
import sklearn


lines = []
with open('./data/driving_log.csv') as csvfile:
    next(csvfile, None)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    for i in range(3):
        # Load images from center, left and right cameras
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = "./data/IMG/" + filename
        image = cv2.imread(local_path)
        images.append(image)

    # Introduce steering correction
    correction = 0.2
    measurement = float(line[3])
    # Steering adjustment for center images
    measurements.append(measurement)
    # Add correction for steering for left images
    measurements.append(measurement+correction)
    # Minus correction for steering for right images
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []

# Augmented data set by adding 'flipped' images 
# so model can learn from reversed images, 
#as well as random brightness 
#(with thanks to Vivek Yadav at http://bit.ly/2kOk6MU for the latter)
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    brightened_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    brightened_image[:,:,2] = brightened_image[:,:,2]*random_bright
    brightened_image = cv2.cvtColor(brightened_image,cv2.COLOR_HSV2RGB)
    flipped_image = cv2.flip(brightened_image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

# Pull the image and steering measurements 
# into NumPy arrays we can use in the model
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Model based on Nvidia's end-to-end architecture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(1,1))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')