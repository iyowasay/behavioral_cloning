import csv
import cv2
import numpy as np
import math
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from keras.optimizers import Adam


def read_lines(csv_path):
	lines = []
	with open(csv_path) as csvfile:
		reading = csv.reader(csvfile)
		for line in reading:
			lines.append(line)
	return lines

def load_images(all_data, correction, split):
	images, measurements = [], []
	lr_images, lr_steer = [], []
	steer_correction = correction # parameter to tune ---------------
	for d in range(len(all_data)):
		lines = read_lines(all_data[d] + '/driving_log.csv')
		image_path = all_data[d] + '/IMG/'
		for line in lines:
			cur_measure = float(line[3]) # steering angle
			img_name = line[0].split('/')[-1]
			bgr = cv2.imread(image_path + img_name) # the input images are jpg images(BGR)
			image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
			images.append(image)
			measurements.append(cur_measure)
			# augmentation - flip the image
			image_flipped = cv2.flip(image, 1)# image_flipped = np.fliplr(image)
			images.append(image_flipped)
			measurements.append(-cur_measure)

			# get left and right images
			left_name = line[1].split('/')[-1]
			right_name = line[2].split('/')[-1]
			rgb_l = cv2.cvtColor(cv2.imread(image_path + left_name), cv2.COLOR_BGR2RGB)
			rgb_r = cv2.cvtColor(cv2.imread(image_path + right_name), cv2.COLOR_BGR2RGB)
			lr_images.append(rgb_l)
			lr_steer.append(cur_measure+steer_correction*0.05*np.random.random())
			lr_images.append(rgb_r)
			lr_steer.append(cur_measure-steer_correction*0.05*np.random.random())


	print('Total images: ', len(images))

	train_x, valid_x, train_y, valid_y = train_test_split(images, measurements, test_size=split, random_state=40)

	train_x += lr_images
	train_y += lr_steer
	assert len(train_x) == len(train_y)

	return train_x, valid_x, train_y, valid_y 

def augmentation(x, y ,activate=False):
	# image augmentation generator 
	if activate==True:
		temp_x, temp_y = [], []
		# tf.keras.preprocessing.image.ImageDataGenerator
		data_aug = ImageDataGenerator(rotation_range=12, shear_range=7, zoom_range=0.2,channel_shift_range=15)
		for i in range(len(y)):
			target = x[i] # the image that need to be augmented
			t1 = np.expand_dims(target, 0)
			iterator = data_aug.flow(t1, batch_size=1)
			batch = iterator.next()
			image = batch[0].astype('uint8')
			temp_x.append(image)
			temp_y.append(y[i])

		X_aug = np.concatenate((x, temp_x))
		y_aug = np.concatenate((y, temp_y))
		return X_aug, y_aug
	else:
		return x, y 
	

def generator(input_images, labels, batch_size=64):
	# generate the next training batch
    num_samples = len(input_images)
    while True: # Loop forever so the generator never terminates
        shuffle(input_images, labels)
        for offset in range(0, num_samples, batch_size):
            batch_x = input_images[offset:offset+batch_size]
            batch_y = labels[offset:offset+batch_size]
            X_train = np.array(batch_x)
            y_train = np.array(batch_y)
            yield shuffle(X_train, y_train)


def preprocess():
	# establish and normalize the model
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((60, 22), (0,0))))
	return model


def LeNet():
	model = preprocess()
	model.add(Conv2D(6,(5,5), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(16,(5,5), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model


def Nvidia():
	# Nvidia end-to-end architechture 
	# source: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
	beta = 0.0001
	l2_regu = False
	model = preprocess()
	if l2_regu == True:
		model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(beta)))
		# model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Dropout(0.4))
		model.add(Flatten())
		model.add(Dense(1164, activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Dropout(0.6))
		model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(beta)))
		model.add(Dense(1))
	else:
		model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
		model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
		model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(Dropout(0.3))
		model.add(Flatten())
		model.add(Dense(1164, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(1))
	return model


# start of model -----------------------------------------------------------------
print('Behavioral Cloning Project:')

steer_correct = 0.2
split_percent = 0.2 # the percentage of validation set
all_data_paths = ['./recover1', './data1','./data2', './data3', './data4', './data5', './data8', './data9']
start_time = time.time()
X_train, X_valid, y_train, y_valid = load_images(all_data_paths, steer_correct, split_percent)# note that the type of these above outputs are lists!
print('Data loading time(s): ', time.time()-start_time)
print('Number of images(before augmentation): '+ str(len(X_train)))
print('Number of validation samples: ', len(X_valid))

# X_aug, y_aug = augmentation(X_train, y_train) - data augmentation is not used.

learning_rate = 0.0001
batch_size = 64
number_of_epochs = 5
n_train = len(X_train)
n_valid = len(X_valid)
next_train = generator(X_train, y_train, batch_size=batch_size)
next_valid = generator(X_valid, y_valid, batch_size=batch_size)
model = Nvidia()
model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
print(model.summary())
history_object = model.fit_generator(next_train, steps_per_epoch=math.ceil(n_train/batch_size), 
	validation_data=next_valid, validation_steps=math.ceil(n_valid/batch_size), 
	epochs=number_of_epochs, verbose=1)

model.save('model.h5')
print('Model saved.')
print()




