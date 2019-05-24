from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Conv2D, Flatten, Dropout, MaxPooling2D, UpSampling2D, GlobalAveragePooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import skimage
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

img_dir = "images"
test_case_file = "./test_case.csv"

batch_size = 128

file_names = list()
for f in os.listdir(img_dir):
    if f.endswith(".jpg"):
        file_names.append(os.path.join(img_dir, f))
num_files = len(file_names)
file_names.sort()

train_x = list()
for f in file_names:
    train_x.append(skimage.io.imread(f, as_gray=False).reshape((32,32,3)))

train_x = np.array(train_x) / 255

input_img = Input(shape=(32, 32, 3))
adam = Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
#x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='latent')(x)


x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#x = BatchNormalization()(x)
#x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=adam, loss='mse')

print(autoencoder.summary())

es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=5,
                   verbose=0)
autoencoder.fit(train_x,
                train_x,
                batch_size=batch_size,
                shuffle=True,
                epochs=1000,
                validation_split=0.1,
                verbose=1,
                callbacks=[es]
                )

autoencoder.save("encoder.h5")



    