'''
python extract_feature.py train/test input_file
'''
import numpy as np
import sys
import csv


def parsing():   
    f1 = open("../../X_train")
    f2 = open("../../Y_train")
    train_x = list(csv.reader(f1))
    train_y = list(csv.reader(f2))
    height = len(train_y)
    width = len(train_x[0])
    train_zeros = list()
    zero_count = 0
    train_ones = list()
    one_count = 0
    for i in range(height):
        if train_y[i][0] == '0':
            train_zeros.append(train_x[i])
            zero_count += 1
        else:
            train_ones.append(train_x[i])
            one_count += 1
    zeros = np.array(train_zeros).flatten().reshape(zero_count, width).astype(int)
    ones = np.array(train_ones).flatten().reshape(one_count, width).astype(int)
    print(zeros.shape, ones.shape)
    np.save("zeros.npy", zeros)
    np.save("ones.npy", ones)


def parsing_test(input_file):
    raw_data = np.genfromtxt(input_file, delimiter=',')
    test_x = raw_data[1: : ]
    np.save("gen_model/test_feature.npy", test_x)


if sys.argv[1] == 'train':
    parsing()
elif sys.argv[1] == 'test':
    parsing_test(sys.argv[2])
else:
    print("python extract_feature.py train/test input_file")
