
###Teaching a car to drive with deep learning

This article is a write up of David Silver's [video tutorial](https://www.youtube.com/watch?v=rpxZ87YFg0M&index=3&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P) for Udacity's excellent [Self-Driving Car Nanodegree](https://classroom.udacity.com/nanodegrees/nd013/syllabus). With thanks to David and all the instructors and mentors on the course.

[//]: # (Image References)

[image1]: ./first_take.jpg "First take"
[image2]: ./first_take_drive.jpg "First take"
[image3]: ./second_take.jpg "Second take"
[image4]: ./second_take_drive.jpg "Second take"
[image5]: ./third_take.jpg "Third take"
[image6]: ./fourth_take.jpg "Fourth take"
[image7]: ./fifth_take_drive.jpg "Fifth take"

First of all, we load the data from the csv file. The images are in three colums, snapped from center, right and left cameras. They have a corresponding steering angle -- the steer the car had to make in that frame to stay on the track. We load both into ```images[]`` and ```measurements``` arrays.

```
import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	source_path = line[0]
	tokens = source_path.split('/')
	filename = tokens[-1]
	local_path = "./data/IMG/" + filename
	image = cv2.imread(local_path)
	images.append(image)
	measurement = line[3]
	measurements.append(measurement)

X_train = np.array(images[1:])
y_train = np.array(measurements[1:])

```

Even at this early stage, with just nine more lines of code, we can add a Keras deep learning model and check the results. 

```
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')
```

This is a simple iteration, with a flattening layer and one fully-connected layer. Of course this doesn't provide much learning for the program, with predictably unsafe driving results:

First model results
Notice the model, running with 10 epochs, begins overfitting by epoch 8 (the validation error starts increasing).

![alt text][image1]

First Autonomous Drive

![alt text][image2]

To begin to make the model more sophisticated, we add a lambda function to normalize the images, this should reduce processing time.

```
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
```
Second take results
Our errors are decreasing, however the overfitting begins again after epoch 7.

![alt text][image3]

And we still drive into the lake

![alt text][image4]

Here's where we start to make the model more sophisticated. We add convolutional layers based on Yann LeCun's LeNet architecture, which includes two convolutional layers, two pooling layers, a flatten layer and three fully-connected layers. We also reduce the epochs to 5 as this may be sufficient to reach a low error without proceeding to overfit. 
The updated model:

```
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
```

We see improved accuracy:

![alt text][image5]

The car, however, is still driving for the lake. Now that we have a reliable model, we can experiment with ways to make the most of our image data. 
Firstly, we need to address the bias that comes from driving the car anti-clockwise around the track. This means that we are mostly adjusting steering to the left, and probably explains the car's eagerness to careen off the track towards the pond in that direction. 
To counter this, we can flip the images. This doubles the image set to 12,856, with around 2,500 split for validation. 

Here's the code:

```
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images[])
y_train = np.array(augmented_measurements[])
```

Fourth take:

![alt text][image6]

Notice the validation is now the lowest we have achieved so far. At this stage, the car is staying on the track longer, however we need to improve the accuracy of its steering further. 
Another bias the car has learned is that it is often driving in the middle of the track and simply going straight. If we're to train the program, we need to make images and related steering angles of turning for corners more prominent in its learning. So we go back to the simulator, and record data of the car steering away from corners. A good way to do this is to drive close to the turn, and only hit record when steering away from the corners. This way the data isn't muddled by the car learning to turn towards the corners. 

Furthermore, we separate the image data into center, left and right cameras. We add a 0.2 steering angle correction for left images, and subtract the same amount for right images.
Our image set is now at 38,572 images, having tripled the 12,000+. Our training may take longer however we should see a leap in accuracy. 

Here's how things look at this stage:

```
import csv
import cv2
import numpy as np

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
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')

```

The car is now learning to keep to the middle and take corners competently.

![alt text][image7]

Looking closely at the image, we see the sky makes about 70 pixels at the top of the frame, and the dashboard about 25 pixels at the bottom. We can crop this to improve processing time and have the model focus only on the essential features. This can be done within the Keras model using their ```Cropping2D``` feature.

```
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(1,1))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
```

At this stage, the loss is down to 0.0157 after just 3 epochs. The car will probably be driving decently, however will likely go off the track at some of the trickier parts of the course. Now we can pay attention to these difficulties, add a few more recordings of the car navigating the features well (such as the bridge, or corners without the usual boundary markings).

We can also have some fun experimenting with different model architectures. ResNet, AlexNet and others spring to mind as possible convolutional models. Below is an archietcture based on Nvidia's [end-to-end](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) deep learning model. 

```
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
```

The behavioral cloning project is a great lesson in the value, necessity and fun of experimentation with deep learning. Altering model architecture and parameters help us build an intuition of how convolutions, drop out layers and subsamples work together to make a useful model. 