import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

x = np.load("x.npy")
x_reshape = x.reshape((28709, 48, 48, 1))
y = np.load("y.npy")
y_cat = to_categorical(y)

train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
valid_datagen = ImageDataGenerator()

train_datagen.fit(x_reshape[2870: , : , : , :])
valid_datagen.fit(x_reshape[0:2870 , : , : , :])

test_generator = valid_datagen.flow(x_reshape[0:2870 , : , : , :], y_cat[0:2870, :], batch_size=32)
train_generator = train_datagen.flow(x_reshape[2870: , : , : , :], y_cat[2870:, :], batch_size=32)


if sys.argv[1] != "new":
    model = load_model(sys.argv[1])
    filepath = "model/m-{epoch:02d}-{categorical_accuracy:.3f}-{val_categorical_accuracy:.3f}.h5"
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train_generator,
                    steps_per_epoch=len(x_reshape) / 32,
                    epochs=1200, 
                    validation_data=test_generator,
                    validation_steps=800,
                    callbacks=[checkpoint])
    model.save("model.h5")
    exit()

model = Sequential() 
#1st convolution layer
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.2))
 
#3rd convolution layer
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
 
model.add(Dense(7, activation='softmax'))
adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

#filepath = "model/m-{epoch:02d}-{categorical_accuracy:.3f}-{val_categorical_accuracy:.3f}.h5"
#checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#model.fit(x_reshape, y_cat, batch_size=16, epochs=100, validation_split=0.1, callbacks=[checkpoint])
model.fit_generator(train_generator,
                    steps_per_epoch=len(x_reshape) / 32,
                    epochs=200, 
                    validation_data=test_generator,
                    validation_steps=100,
                    #callbacks=[checkpoint]
                    )
model.save("model.h5")