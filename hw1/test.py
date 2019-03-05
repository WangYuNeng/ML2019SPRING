'''
python test.py output_file
'''

import numpy as np
import sys


if len(sys.argv) != 2:
    print("python test.py output_file")
    exit()

w = np.load("weight.npy")
test_x = np.load("test_feature.npy")
test_y = test_x.dot(w) 
with open(sys.argv[1], 'w') as f:
    f.write("id,value\n")
    i = 0
    for array in test_y:
        if array[0] <= 0:
            array[0] = 0
        line = "id_" + str(i) + "," + str(array[0]) + '\n'
        f.write(line)
        i += 1