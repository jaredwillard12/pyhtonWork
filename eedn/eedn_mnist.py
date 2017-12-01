#import sys
#import os 

#dir_path = os.path.dirname(os.path.abspath("eedn_mnist_notebook"))
#sys.path.insert(0, dir_path)
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from eedn_layers import eednLayer, EednPrediction

batch_size = 128
num_classes = 10
epochs = 2000

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train -= 128
x_test -= 128
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

eednLayer(model, 12, kernel_size = 3,strides = 1, input_shape=input_shape, transduction=True, padding = 'same')
eednLayer(model, 252, kernel_size = 4, groups= 2,strides = 2, padding = 'same')
eednLayer(model, 256, kernel_size = 1, groups= 2,strides = 1)
eednLayer(model, 256, kernel_size = 2, groups= 8,strides = 2)
eednLayer(model, 512, kernel_size = 3, groups= 32,strides = 1, padding = 'same')
eednLayer(model, 512, kernel_size = 1, groups= 4,strides = 1)
eednLayer(model, 512, kernel_size = 1, groups= 4,strides = 1)
eednLayer(model, 512, kernel_size = 1, groups= 4,strides = 1)
eednLayer(model, 512, kernel_size = 2, groups= 16,strides = 2)
eednLayer(model, 1024, kernel_size = 3, groups= 64,strides = 1, padding = 'same')
eednLayer(model, 1024, kernel_size = 1, groups= 8,strides = 1)
eednLayer(model, 1024, kernel_size = 2, groups= 32,strides = 2)
eednLayer(model, 1024, kernel_size = 1, groups= 8,strides = 1)
eednLayer(model, 1024, kernel_size = 1, groups= 8,strides = 1)
eednLayer(model, 2048, kernel_size = 1, groups= 8,strides = 1)
model.add(EednPrediction(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(lr=20),
              optimizer=keras.optimizers.SGD(momentum=0.9,lr=20),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
