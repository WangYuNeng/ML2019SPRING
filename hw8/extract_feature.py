import numpy as np
import sys
import csv


def parsing(input_file):
    f = open(input_file, "r")
    raw = f.read().strip().replace(',', ' ').split('\n')[1:]
    temp = ' '.join(raw)
    raw_data = np.array(temp.split()).astype('float').reshape(-1, 48*48+1)
    np.random.shuffle(raw_data)    
    raw_y = raw_data[ : , 0 ]
    raw_x = raw_data[ : , 1: ] / 255
    (height_x, width_x) = raw_x.shape
    reshape_x = raw_x.reshape((height_x, 48, 48))
    print(reshape_x.shape)
    np.save("x.npy", reshape_x)
    np.save("y.npy", raw_y)

def parsing_test(input_file):
    f = open(input_file, "r")
    raw = f.read().strip().replace(',', ' ').split('\n')[1:]
    temp = ' '.join(raw)
    raw_data = np.array(temp.split()).astype('float').reshape(-1, 48*48+1)
    test_x = raw_data[ : , 1: ] / 255
    # extract feature
    (height, width) = test_x.shape
    print(test_x.shape)
    test_x = test_x.reshape((height, 48, 48, 1))
    # normalize
    np.save("test_feature.npy", test_x)


if sys.argv[1] == 'train':
    parsing(sys.argv[2])
if sys.argv[1] == 'test':
    parsing_test(sys.argv[2])