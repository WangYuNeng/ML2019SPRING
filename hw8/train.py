from create_model import *
from transform import load_weight_16
import numpy as np
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":

    x = np.load("x.npy")
    x_reshape = x.reshape((x.shape[0], 48, 48, 1))
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

    test_generator = valid_datagen.flow(x_reshape[0:2870 , : , : , :], y_cat[0:2870, :], batch_size=512)
    train_generator = train_datagen.flow(x_reshape[2870: , : , : , :], y_cat[2870:, :], batch_size=512)

    model = create_model(input_shape=(48, 48, 1),
              alpha=0.5,
              depth_multiplier=1,
              classes=7)
    adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    filepath = "model/m-{epoch:02d}-{categorical_accuracy:.3f}-{val_categorical_accuracy:.3f}.h5"

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    history = model.fit_generator(train_generator,
                    steps_per_epoch=len(x_reshape) / 64,
                    epochs=300, 
                    validation_data=test_generator,
                    validation_steps=100,
                    callbacks=[checkpoint])

