import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D, ZeroPadding2D, AveragePooling2D, BatchNormalization

data_dir = '/home/tim/Datasets/flower_photos'

image_width = 100
image_height = 100
batch_size = 32

data = ImageDataGenerator(rescale=1./255, validation_split=0.1, rotation_range=45, zoom_range=0.2, preprocessing_function=None,
                          horizontal_flip=True, vertical_flip=True, fill_mode='nearest', shear_range=0.1,
                          height_shift_range=0.1, width_shift_range=0.1)

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

train = data.flow_from_directory(directory=data_dir,
                                 target_size=(image_height, image_width),
                                 class_mode='sparse',
                                 subset='training',
                                 batch_size=batch_size)

test = data.flow_from_directory(directory=data_dir,
                                target_size=(image_height, image_width),
                                class_mode='sparse',
                                subset='validation',
                                batch_size=batch_size)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='elu', padding='same', input_shape=(image_width, image_height, 3)))
model.add(MaxPool2D(2))
model.add(AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid'))

model.add(Conv2D(32, (3, 3), padding='same', activation='elu'))
model.add(MaxPool2D(2))
model.add(ZeroPadding2D())
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
model.add(MaxPool2D(2))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='elu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(train, batch_size=32, validation_data=test, epochs=25, verbose=1)

done = input('save: ')
if done == 'yes':
    model.save('/home/tim/trained/flowerReck1.h5')
else:
    pass
