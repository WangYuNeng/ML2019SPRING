import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras import optimizers
from keras import regularizers
from keras.utils import to_categorical


train_data = np.load("train_feature.npy")
(height, width) = train_data.shape
#model = Sequential()
#model.add(Dense(units=100, activation='relu', input_dim=width-1, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
#model.add(Dense(units=80, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
#model.add(Dense(units=1, activation='sigmoid'))
#adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model = load_model("model150.h5")
model.fit(train_data[ : , : -1], train_data[ : , -1:], batch_size=32, epochs=50, validation_split=0.1)


model.save("model200.h5")