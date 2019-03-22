'''
python test.py output_file
'''

import numpy as np
import sys
from numpy.linalg import inv, det
from math import pi, exp
import time


if len(sys.argv) != 2:
    print("python test.py output_file")
    exit()


p_zero = 24720/(24720+7841)
mean_zero = np.load("gen_model/0_mean.npy")
mean_one = np.load("gen_model/1_mean.npy")
cov_mutual = np.load("gen_model/mutual_cov.npy")
test_x = np.load("gen_model/test_feature.npy")



with open(sys.argv[1], 'w') as f:
    f.write("id,label\n")
    i = 1
    for row in test_x:
        X = row - mean_zero
        up1 = -0.5*X.dot(inv(cov_mutual)).dot(X.transpose())
        X = row - mean_one
        up2 = -0.5*X.dot(inv(cov_mutual)).dot(X.transpose())
        ratio  = exp(up2[0,0] - up1[0,0])
        prob = p_zero / (p_zero + (1-p_zero)*ratio)    
        if prob > 0.5:
            label = '0'
        else:
            label = '1'
        line = str(i) + "," + label + '\n'
        f.write(line)
        i += 1