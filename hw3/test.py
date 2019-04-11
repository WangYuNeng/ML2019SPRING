import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

list_file = open(sys.argv[2], 'r')
test_x = np.load("test_feature.npy")
test_y = np.zeros((test_x.shape[0], 7))
while (1):
    line = list_file.readline()
    if line == ""  or line == "\n":
        break
    model = load_model(line[:-1])
    test_y += model.predict(test_x, batch_size=100)
with open(sys.argv[1], 'w') as f:
    f.write("id,label\n")
    i = 0
    for row in test_y:
        max_l = 0
        label = 0
        for j in range(7):
            if row[j] > max_l:
                label = j
                max_l = row[j]
        line = str(i) + "," + str(label) + '\n'
        f.write(line)
        i += 1