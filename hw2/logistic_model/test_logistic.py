'''
python test.py output_file
'''

import numpy as np
import sys


if len(sys.argv) != 2:
    print("python test.py output_file")
    exit()

w = np.load("logistic/weight.npy")
test_x = np.load("logistic/test_feature.npy")
test_y = test_x.dot(w) 
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