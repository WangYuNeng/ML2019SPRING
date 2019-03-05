'''
python extract_feature.py train/test input_file
'''
import numpy as np
import sys


def parsing(input_file):   
    #parsing training data into numpy array
    raw_data = np.genfromtxt(input_file, delimiter=',', encoding='big5')
    train_data = np.empty((18, 0), float)
    (height, width) = raw_data.shape
    for i in range(1, height, 18):
        train_data = np.hstack((train_data, raw_data[i:i+18, 3:]))
    train_data[10, :] = 0.0 # RAINFALL
    train_data[4, :] = 0.0 # NO
    train_data[6, :] = 0.0 # NOx

    # extract feature
    (height, width) = train_data.shape
    train = np.empty((0, 18*9+2), float)
    for i in range(width - 9):
        if i % (20*24) >= (20*24 - 9):
            continue
        append_data = np.append(train_data[ : , i:i+9].flatten(), [1, train_data[9, i+9]]).reshape(1, 18*9+2)
        flag = True
        for num in append_data[0]:
            if num < 0:
                flag = False
        if flag == False:
            continue
        train = np.append(train, append_data, axis=0)

    # normalize
    norm_array = np.empty((0,2))
    for i in range(18*9):
        data = train[ : , i]
        avg = np.average(data)
        std = np.std(data)
        if std != 0:
            train[ : , i] = (data - avg) / std
        else:
            train[ : , i] = data - avg
        norm_array = np.append(norm_array, np.array([avg, std]).reshape(1,2), axis=0)
    np.save("norm.npy", norm_array)
    np.save("train_feature.npy", train)

def parsing_test(input_file):
    norm = np.load("norm.npy")
    raw_data = np.genfromtxt(input_file, delimiter=',', encoding='big5')

    # extract feature
    test_x = np.empty((0, 18*9 + 1), float)
    (height, width) = raw_data.shape
    for i in range(0, height, 18):
        raw_data[i+10, : ] = 0.0 # RAIN_FALL
        raw_data[i+4, : ] = 0.0 # NO
        raw_data[i+6, : ] = 0.0 # NOx
        append_data = np.append(raw_data[i:i+18, 2:].flatten(), [1]).reshape((1, 18*9 + 1))
        j = 0
        for j in range(18*9):
            num = append_data[0][j]
            if num < 0:
                if j == 0:
                    append_data[0][j] = append_data[0][j+1]
                elif j == 18*9-1:
                    append_data[0][j] = append_data[0][j-1]
                else:
                    append_data[0][j] = (append_data[0][j-1] + append_data[0][j+1]) / 2
        test_x = np.append(test_x, append_data, axis=0)

    # normalize
    for i in range(18*9):
        data = test_x[ : , i]
        avg = norm[i, 0]
        std = norm[i, 1]
        if std != 0:
            test_x[ : , i] = (data - avg) / std
        else:
            test_x[ : , i] = data - avg
    np.save("test_feature.npy", test_x)


if len(sys.argv) != 3:
    print("python extract_feature.py train/test input_file")
    exit()

if sys.argv[1] == 'train':
    parsing(sys.argv[2])
elif sys.argv[1] == 'test':
    parsing_test(sys.argv[2])
else:
    print("python extract_feature.py train/test input_file")
