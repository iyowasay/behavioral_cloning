# **Behavioral Cloning Project** 
---
The goals / steps of this project are the following:
* Use the simulator to collect data of "good" driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the track
* Summarize the results with a written report

---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. Here the file `drive.py` will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection. At the end, the vehicle should be able to drive autonomously around the track without leaving the road.

Here is [the bird's-eye view video](https://youtu.be/gHSvIalDYVw) of the final result.

### Preparation

The first and the most important step for end-to-end deep learning is to collect sufficient and useful data. Though one lap of data is enough for the model to successfully drive around the track, the performance of the car can still be improved. Therefore, more data have to be acquired. This can be done by manually driving the car in the opposite direction or applying data augmentation. 
In total, 7 laps of data with center lane driving are used in this project. 

- 4 laps of center lane driving
- 3 laps of center lane driving in the opposite direction 
- 1 lap of recovering from the positions very close to the edge of the road

I use 20% of entire data as validation set, which helps to ensure that the model was not overfitting. Note that it is recommended to split the training and validation data before augmentation. The validation set should remain the best possible measurement in order to get accurate evaluation of the model. Thus applying augmentation on validation data does not make the model better. 

#### GPU v.s. CPU

GPUs are optimized for high-throughput computation, running as many simultanious computation as possible. They are usually used in graphically intensive video games. CPUs, on the other hand, are mostly optimized for latency, running a single thread of instructions as quickly as possible. It turns out that deep learning have a high level of parallelism, which is the strength of GPU. Therefore, one can get a significant improvement of training process by running your network on GPU(roughly 5 times faster than on CPU). In this project, all the training run on the GPU in the Udacity workspace.

### Data augmentation

Due to the default heading of the simulator and the shape of the track, the vehicle often tends to turn left. In order to let the model learn steering properly, we need to obtain more right-turning data and make the dataset more balanced. One thing can be done is to flip the image and add a negative value of steering angle. 

If the vehicle does not turn enough to fix back to the center of the road, we can take advantage of the multicamera system. Add or subtract a correction factor from the center steering angle when processing the left or right images respectively. This correction factor can be determined by simple parameter tuning, or one can apply trigonomotry and physics to get a more precise value.

Here image augmentation techniques, such as rotation, scaling, color channel shifting, are not included, since it makes the processing time too long(`ImageDataGenerator()`). 

### Python Generator

Generator is a great way to cope with large amount of data. Instead of processing all the images at once, the python generator divide the dataset into batches of data and process them on the fly, which makes the training process more memory-efficient. It works like a different thread, running separately from the main steps.

### Model architechture

The architechture of this project is based on the [Nvidia end-to-end self driving car](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), which is a demonstration of transfer learning. RELU activation functions are used to introduce nonlinearity. In addition, the model used an Adam optimizer, so manually tuning the learning rate is not necessary. Also, the model contains dropout layers in order to reduce overfitting. The detailed structure of the model is shown in the following figure.

#### Preprocessing steps
1. Convert BGR(.jpg) to RGB image.
2. Normalize all the pixel values.
3. Crop the upper and lower part of original images, since these pixels don't contain important information for training.

<img src="/examples/table.png" alt="table" width="600" height="600"/>

Parameters:
- Dropout rate = 0.3
- Steer correction = 0.2 + 0.05*np.random.random()
- Learning rate = 0.0001
- Epochs = 5
- Batch size = 64

Statistics:
- Number of layers: 14
- Number of center images: 9173
- Data loading time(s): 35.175
- Number of training images(after augmentation): 33022
- Number of validation samples: 3670


#### Tunning approach

1. Drive manually in the simulator and collect data.
2. Validate the training process and the connection to the simulator by useing LeNet network.
3. Adapt the Nvidia model and add additional layers.
4. Convert color channel of input images into RGB -> model improved!(due to the fact that the images captured from the simulator are BGR(.jpg), whereas `drive.py` load images in RGB to predict the steering angles)
5. Decide to use smaller batch size(higher accuracy but more noisy and time-consuming).
6. Tune hyperparameters -> without falling off the track but shaky.
7. Add randomness to the correction factor of steering angle -> model improved!
8. Try different combination of datasets(since the quality may vary between each sets).
9. Save the BEST model!

### Result and reflection

Current trained model `model.h5` is able to drive around the track several times without falling off, with the set speed of 25. The training results are shown below. 

Epoch 1/5
516/516 [==============================] - 65s 126ms/step - loss: 0.0320 - val_loss: 0.0373
Epoch 2/5
516/516 [==============================] - 62s 121ms/step - loss: 0.0295 - val_loss: 0.0343
Epoch 3/5
516/516 [==============================] - 62s 121ms/step - loss: 0.0283 - val_loss: 0.0329
Epoch 4/5
516/516 [==============================] - 62s 121ms/step - loss: 0.0273 - val_loss: 0.0312
Epoch 5/5
516/516 [==============================] - 62s 121ms/step - loss: 0.0263 - val_loss: 0.0305

Here the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Note that the mean squared errors(losses) between each training attempts are not comparable, meaning that smaller loss does not correspond to better performance.  

The quality and amount of collected data play important roles in end-to-end deep learning. For instance, the model might not know how to react while facing rare situation or edge cases that are not included in the training data. 
Moreover, the performance also depends on the targeting speed from drive.py. It is easier to achieve a complete drive-around with lower set speed. 

### Ways to improvement

1. Give the control back to the driver(manual driving) and start data recording as soon as it is about to drive off the track. These data will help the model learn how strong the vehicle should steer at the edge of the road.

2. Get more data from the vehicle system, not just the images. Better time series data or other information from the simulator( in reality, signals of other kinds of sensor readings) may help while training the model.

3. Instead of predicting the steering angle, predict where the vehicle should be placed in each frame and then to move or steer according to this prediction. It might be closer to the behavior of human drivers. 

4. Add image augmentation using OpenCV for generalization and increasing robustness.

<!-- Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 78, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 37, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              7376268   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 7,629,687
Trainable params: 7,629,687
Non-trainable params: 0  -->


