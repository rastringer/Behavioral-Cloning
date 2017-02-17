
[//]: # (Image References)

[image1]: ./data/IMG/center_2016_12_01_13_30_48_287.jpg "Simulator"
[image2]: ./images/model.png "Convolutional Neural Network in Keras"
[image3]: ./images/car_driving.png "Autonomous Mode"

####Deep learning to teach cars how to drive

###Image processing and augmentation

![alt text][image1]

I applied various techniques to make processing the images through the model faster and to improve accuracy. 
Firstly, we load the three colummns of images from the center, left and right cameras. I then tripled the image data set using two augmentative techniques. The first randomly adjusted the brightness, which should improve the algorithm's ability to identify patterns even in parts of the test track where there are, for example, shadows from trees that obscure the lane lines. 
Another augmentation was to reverse, or flip, the images. One potential shortcoming of data drawn from a car driving anti-clockwise on a track is that it typically has to adjust its position with left turns. This can cause a bias for left leaning, much like a supermarket trolley with a kinked wheel. 

I adjusted steering values for left and right images to improve movement from the left and right of the track towards the center.

Since the bottom ~25 pixels of the image are of the car's front, and the top ~70 pixels are of the sky, I cropped the images to focus processing on the road. The crop is integral to the model, using Keras' Cropping2D feature.

###Architecture

I based the end-to-end convolutional neural network architecture summarized by Nvidia here: http://bit.ly/1T206A2.
The network consists of 12 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. 
Firstly, we use Keras' lambda feature to normalize the images. Then we crop out the front of the car in the bottom of the image and the sky from the top, using the Cropping2D method.
The model's first 3 convolutional layers have a 2 X 2 stride, 5 X 5 kernel and a relu activation. The last 2 convolutional layers are non-strided, with a 3 X 3 kernel size. Following the convolutional layers, we have a flatten layer, followed by the 3 fully-connected layers featuring 100, 50 and finally 1 neuron.

![alt text][image2]

###Fine Tuning

It was illustrative to try the model on both Udacity's data and my own, which was collected from driving in the simulator. I tried several iterations of controlling the car for 2 laps, one anti-clockwise and another clockwise to remove any directional bias, followed by a series of shorter recordings where I would sway to the ride of the road and record the recovery. 

Once the model was working respectably, I decided to focus on improving it with the Udacity data. I felt there may be a temptation to focus on better simulator driving to cure model ills, and while those recordings are of course vital, I wanted to focus on the processing and convolutional neural network.

![alt text][image3]