'''
python train.py
'''

import numpy as np
import math
import sys


def first_order(train_x, train_y):
    # initialize 
    w = np.ones((107, 1)) # including bias
    #w = np.load("weight.npy")
    iteration = 1000
    lr = 0.1
    lr_w = np.zeros((107, 1)) + 0.00000000000000000001 # prevent from divided by 0
    
    # training
    for i in range(iteration):
        '''
        loss = (ty - (w*x + b)) ** 2
        grad_w = -2 * (ty - (w*x + b)) * x
        '''
        y = 1/(1+np.exp(-1*train_x.dot(w)))
        loss_sqrt = train_y - y
        grad_w = -1 * train_x.transpose().dot(loss_sqrt)

        lr_w += grad_w ** 2
        w -= lr / np.sqrt(lr_w) * grad_w

        if i % 10 == 0:
            correct = 0
            j = 0
            for label in y:
                if label[0] > 0.5:
                    num = 1
                else:
                    num = 0
                if num == train_y[j]:
                    correct += 1
                j += 1
            print(i, correct/32861, loss_sqrt.sum()/32861)
    np.save("weight.npy", w)


train_data = np.load("train_feature.npy")
first_order(train_data[ : , : -1], train_data[ : , -1:])