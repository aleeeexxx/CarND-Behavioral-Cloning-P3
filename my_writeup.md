**Behavioral Cloning Project**
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

           
[//]: # (Image References)

[image1]: ./examples/center_2016_12_01_13_30_48_287.jpg "Center Image"
[image2]: ./examples/center_2020_03_01_07_37_14_132.jpg "Recovery Image"
[image3]: ./examples/center_2020_03_01_07_47_18_682.jpg "Recovery Image"
[image4]: ./examples/msel.png "MSE"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a convolutional neural network, obtain data to feed in the network, evaluate the performance and adjust the model or dataset to enhance the neural network.

My first step was to use the convolution neural network model published by the autonomous vehicle team at Nvidia, I thought this model might be appropriate because the model has been proved to be useful and robust enough.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I record video in a new folder, only focusing on where those failures occurred. Then using the new data to retrain the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 81-94) consisted of a convolution neural network starting with a normalization layer before images being cropped to only keep the road area, followed by 5 convolutional layers and finally followed by 3 fully connected layers.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the sample data provided by Udacity that already existed in the workspace. Here is an example image of center lane driving:

![alt text][image1]

But the autonomous vehicle always tends to drive off the track at a certain tricky location. For example, the intersection between the track and parkway or sharp corner. So I create the data2 folder to focus on those challenges, recording only at those positions and starting at the edge of the track then drive to the center. These images are shown below:

![alt text][image2]
![alt text][image3]


Then I repeated this process, recording data3 and train the model again to fix the remaining problems, after the collection process, I had 4243 and 1335 data points in data2 and data3 respectively. I then preprocessed this data by flip the images horizontally, not only collect twice as much data as before but also make the model generalize better. Then I add a normalization layer, scaling the range of data to -0.5 to 0.5 and passing to the convolutional layers and fully connected layers. I used mean squared error as a loss function since it is a regression problem and finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model.
The validation set helped determine if the model was over or underfitting. The ideal number of epochs changes with the data folder I choose to train. 2 is what works for folder data3. The loss for each epoch is shown below:
![alt text][image4]

Finally, I used an adam optimizer so that manually training the learning rate wasn't necessary.
