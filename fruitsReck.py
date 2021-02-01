import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, BatchNormalization


datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

batch_size = 100
img_width = 64
img_height = 64

train = '/home/tim/Datasets/fruitImages/train'

train_ds = datagen.flow_from_directory(directory=train,
                                       target_size=(img_width, img_height),
                                       subset='training',
                                       class_mode='sparse',
                                       batch_size=batch_size)

val_ds = datagen.flow_from_directory(directory=train,
                                     target_size=(img_width, img_height),
                                     class_mode='sparse',
                                     subset='validation',
                                     batch_size=batch_size)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(33, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1, batch_size=60)

done = input('save: ')
if done == 'yes':
    model.save('/home/tim/trained/fruitsReck1.h5')
else:
    pass
