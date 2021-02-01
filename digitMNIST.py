import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPool2D, Conv2D
from tensorflow.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plt.matshow(x_train[44], y_train[44])
# plt.show()

# x_train = x_train.reshape(60000, 784)
x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 784)
x_test = x_test.reshape(10000, 28, 28, 1)


x_train = x_train/255.0
y_train = y_train/255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
# model.add(Dense(128, activation='relu'))
model.add(Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=1, validation_split=0.2)

model.save('/home/tim/trained/digit_recognition1.h5')



'''
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('/home/tim/trained/digit_recognition.h5')
img = '/home/tim/Pictures/testdigit.jpg'
img = np.reshape(28, 28)
model.predict(img)
'''
