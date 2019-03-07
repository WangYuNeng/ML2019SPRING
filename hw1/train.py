'''
python train.py
'''

import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import optimizers


train_data = np.load("train_feature.npy")
np.random.shuffle(train_data)
model = Sequential()
#model.add(Dense(units=1, activation='linear', input_dim=18*9))
model.add(Dense(units=1, activation='linear', input_dim=18*9))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#adagrad = optimizers.Adagrad(lr=10, epsilon=None, decay=0.0)
#earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=0, verbose=0, mode='auto', baseline=31, restore_best_weights=True)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(train_data[ : , : -2], train_data[ : , -1:], batch_size=32, epochs=2000, validation_split=0.1) # validation_split
model.save("model.h5")