import tensorflow as tf

# import dataset

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 60000 train
# 10000 test
# 28 by 28 images

# plot example

from matplotlib import pyplot as plt
plt.imshow(x_train[0], cmap = 'binary')
plt.show()

# one hot encoding

from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# preprocessing

import numpy as np
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

# display pixel values
# print(set(x_train_reshaped[0]))

# data normalization
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

# creating model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
	Dense(128, activation='relu', input_shape=(784,)),
	Dense(128, activation='relu'), 
	Dense(10, activation='softmax')
	])

model.compile(
	optimizer = 'sgd',
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
	)

model.summary()

# training the model

model.fit(x_train_norm, y_train_encoded, epochs = 3)

# evaluate the model

loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)

# make predictions

preds = model.predict(x_test_norm)

# plot results

plt.figure(figsize = (12, 12))

start_index = 0

for i in range(25):
	plt.subplot(5, 5, i+1)
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	pred = np.argmax(preds[start_index+i])
	# gt = ground truth
	gt = y_test[start_index+i]
	col = 'g'
	if pred != gt:
		col = 'r'

	plt.xlabel('i = {}, pred = {}, gt = {}'.format(start_index+i, pred, gt), color = col)
	plt.imshow(x_test[start_index+i], cmap = 'binary')

plt.show()

# look at specific index

plt.plot(preds[8])
plt.show()
