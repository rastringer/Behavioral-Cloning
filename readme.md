
Deep learning to teach cars how to drive

Image processing

I processed the images from the Udacity data set using random brightness highlighting and a function to crop the hood and sky from images. This focus our field of view on the road ahead and should improve accuracy. After reading of the advantages of collecting data in the simulator using a joystick and lamenting my lack of such a possession, I used decided to exclusively use the Udacity data.

I adjusted steering values for left and right images to improve movement from the left and right of the track towards the center.

Architecture

I employed the end-to-end convolutional neural network architecture summarized by Nvidia here: http://bit.ly/1T206A2
The model consists of five convolutional layers (with a 2 x 2 stride and 5 x 5 kernel) and three fully connected layers with zero strides and a 3 x 3 kernel size. 

Fine Tuning

I achieved best results by using the Udacity data over my own, captured by using my keyboard to control the car in the simulator. I began with RELU, ELU, and in the end elected to use the LeakyRELU after improved results.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
Normalization (Lambda)           (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
Conv1 (Convolution2D)            (None, 64, 64, 24)    1824        Normalization[0][0]              
____________________________________________________________________________________________________
LeakyRelu1 (LeakyReLU)           (None, 64, 64, 24)    0           Conv1[0][0]                      
____________________________________________________________________________________________________
Conv2 (Convolution2D)            (None, 60, 60, 36)    21636       LeakyRelu1[0][0]                 
____________________________________________________________________________________________________
LeakyRelu2 (LeakyReLU)           (None, 60, 60, 36)    0           Conv2[0][0]                      
____________________________________________________________________________________________________
MaxPool1 (MaxPooling2D)          (None, 30, 30, 36)    0           LeakyRelu2[0][0]                 
____________________________________________________________________________________________________
Dropout_0.5_1 (Dropout)          (None, 30, 30, 36)    0           MaxPool1[0][0]                   
____________________________________________________________________________________________________
Conv3 (Convolution2D)            (None, 30, 30, 48)    43248       Dropout_0.5_1[0][0]              
____________________________________________________________________________________________________
Leaky Relu 3 (LeakyReLU)         (None, 30, 30, 48)    0           Conv3[0][0]                      
____________________________________________________________________________________________________
Conv4 (Convolution2D)            (None, 28, 28, 64)    27712       Leaky Relu 3[0][0]               
____________________________________________________________________________________________________
Leaky Relu 4 (LeakyReLU)         (None, 28, 28, 64)    0           Conv4[0][0]                      
____________________________________________________________________________________________________
MaxPool2 (MaxPooling2D)          (None, 14, 14, 64)    0           Leaky Relu 4[0][0]               
____________________________________________________________________________________________________
Dropout_0.5_2 (Dropout)          (None, 14, 14, 64)    0           MaxPool2[0][0]                   
____________________________________________________________________________________________________
Conv5 (Convolution2D)            (None, 12, 12, 64)    36928       Dropout_0.5_2[0][0]              
____________________________________________________________________________________________________
Leaky Relu 5 (LeakyReLU)         (None, 12, 12, 64)    0           Conv5[0][0]                      
____________________________________________________________________________________________________
MaxPool3 (MaxPooling2D)          (None, 6, 6, 64)      0           Leaky Relu 5[0][0]               
____________________________________________________________________________________________________
Dropout_0.5_3 (Dropout)          (None, 6, 6, 64)      0           MaxPool3[0][0]                   
____________________________________________________________________________________________________
Flatten (Flatten)                (None, 2304)          0           Dropout_0.5_3[0][0]              
____________________________________________________________________________________________________
Dense512 (Dense)                 (None, 512)           1180160     Flatten[0][0]                    
____________________________________________________________________________________________________
Leaky Relu 6 (LeakyReLU)         (None, 512)           0           Dense512[0][0]                   
____________________________________________________________________________________________________
Dropout_0.5_4 (Dropout)          (None, 512)           0           Leaky Relu 6[0][0]               
____________________________________________________________________________________________________
Output (Dense)                   (None, 1)             513         Dropout_0.5_4[0][0]              
================================================================================================
