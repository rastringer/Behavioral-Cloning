
Deep learning to teach cars how to drive

Image processing

I processed the images from the Udacity data set using random brightness highlighting and a function to crop the hood and sky from images. This focus our field of view on the road ahead and should improve accuracy. After reading of the advantages of collecting data in the simulator using a joystick and lamenting my lack of such a possession, I used decided to exclusively use the Udacity data.

I adjusted steering values for left and right images to improve movement from the left and right of the track towards the center.

Architecture

I employed the end-to-end convolutional neural network architecture summarized by Nvidia here: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf 
The model consists of five convolutional layers (with a 2 x 2 stride and 5 x 5 kernel) and three fully connected layers with zero strides and a 3 x 3 kernel size. 

Fine Tuning

I achieved best results by using the Udacity data over my own, captured by using my keyboard to control the car in the simulator. I began with RELU, ELU, and in the end elected to use the LeakyRELU after improved results.

