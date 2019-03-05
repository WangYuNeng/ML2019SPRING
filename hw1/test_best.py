'''
python test.py output_file
'''

import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers


if len(sys.argv) != 2:
    print("python test.py output_file")
    exit()

model = load_model("model.h5")
test_x = np.load("test_feature.npy")
test_y = model.predict(test_x[: , : -1], batch_size=100)
with open(sys.argv[1], 'w') as f:
    f.write("id,value\n")
    i = 0
    for array in test_y:
        if array[0] <= 0:
            array[0] = 0
        line = "id_" + str(i) + "," + str(array[0]) + '\n'
        f.write(line)
        i += 1