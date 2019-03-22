import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers


if len(sys.argv) != 2:
    print("python test.py output_file")
    exit()

model = load_model("best_model/model.h5")
test_x = np.load("best_model/test_feature.npy")
test_y = model.predict(test_x, batch_size=100)
with open(sys.argv[1], 'w') as f:
    f.write("id,label\n")
    i = 1
    for array in test_y:
        if array[0] >0.5:
            label = '1'
        else:
            label = '0'
        line = str(i) + "," + label + '\n'
        f.write(line)
        i += 1