'''
python extract_feature.py train/test input_file
'''
import numpy as np
import sys


def parsing():   
    raw_x = np.genfromtxt("../../X_train", delimiter=',')
    raw_y = np.genfromtxt("../../Y_train").reshape(32561, 1)
    bias = np.ones((32561,1))
    raw_x = np.hstack((raw_x, bias))
    train_data = np.hstack((raw_x, raw_y))
    (height, width) = train_data.shape
    norm_array = np.empty((0,2))
    for i in range(width-2):
        data = train_data[ : , i]
        avg = np.average(data)
        std = np.std(data)
        if std != 0:
            train_data[ : , i] = (data - avg) / std
        else:
            train_data[ : , i] = data - avg
        norm_array = np.append(norm_array, np.array([avg, std]).reshape(1,2), axis=0)
    np.save("norm.npy", norm_array)
    np.save("train_feature.npy", train_data)

def parsing_test(input_file):
    norm = np.load("logistic_model/norm.npy")
    raw_data = np.genfromtxt(input_file, delimiter=',')
    test_x = raw_data[1: : ]
    # extract feature
    (height, width) = test_x.shape
    test_x = np.hstack((test_x, np.ones((height, 1))))
    # normalize
    for i in range(width-1):
        data = test_x[ : , i]
        avg = norm[i, 0]
        std = norm[i, 1]
        if std != 0:
            test_x[ : , i] = (data - avg) / std
        else:
            test_x[ : , i] = data - avg
    np.save("logistic_model/test_feature.npy", test_x)


if sys.argv[1] == 'train':
    parsing()
elif sys.argv[1] == 'test':
    parsing_test(sys.argv[2])
else:
    print("python extract_feature.py train/test input_file")
