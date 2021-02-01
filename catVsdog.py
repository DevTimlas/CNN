import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

test_data_dir = '/home/tim/Datasets/catVdogDataset/test_set'
train_data_dir = '/home/tim/Datasets/catVdogDataset/training_set'

img_width = 180
img_height = 180
batch_size = 100

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
                                              target_size=(img_width, img_height),
                                              classes=['dogs', 'cats'],
                                              class_mode='binary',
                                              batch_size=batch_size)

validation_generator = datagen.flow_from_directory(directory=test_data_dir,
                                                   target_size=(img_width, img_height),
                                                   classes=['dogs', 'cats'],
                                                   class_mode='binary',
                                                   batch_size=batch_size)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
                 input_shape=(img_width, img_height, 3)))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid')) #sigmoid can only predict 0 or 1

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=5)
done = input('save?:  ')
if done == "yes":
    model.save('/home/tim/trained/catVsdog1.h5')
else:
    pass

