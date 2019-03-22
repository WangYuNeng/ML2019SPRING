import numpy as np
from numpy.linalg import inv, det
from math import pi, exp

width = 106

train_zero = np.load("zeros.npy")
mean_zero = np.zeros((1,width))
cov_zero = np.zeros((width, width))
(height_zero, w) = train_zero.shape

train_one = np.load("ones.npy")
mean_one = np.zeros((1,width))
cov_one = np.zeros((width, width))
(height_one, w) = train_one.shape

print(height_zero, height_one)

p_zero = height_zero / (height_zero + height_one)
cov_mutual = np.zeros((width, width))

for i in range(width):
    mean_zero[ 0, i] = np.average(train_zero[ : , i])
    mean_one[ 0, i] = np.average(train_one[ : , i])

for row in train_zero:
    X = row - mean_zero
    cov_zero += X.transpose().dot(X)
for row in train_one:
    X = row - mean_one
    cov_one += X.transpose().dot(X)

cov_mutual = (cov_zero + cov_one) / (height_zero + height_one)
cov_det = det(cov_mutual)
'''
error = 0
count = 0
for row in train_zero:
    count += 1
    X = row - mean_zero
    up1 = -0.5*X.dot(inv(cov_mutual)).dot(X.transpose())
    X = row - mean_one
    up2 = -0.5*X.dot(inv(cov_mutual)).dot(X.transpose())
    ratio  = exp(up2[0,0] - up1[0,0])
    label = p_zero / (p_zero + (1-p_zero)*ratio)
    if label < 0.5:
        error += 1
        print(error/count)

for row in train_one:
    count += 1
    X = row - mean_zero
    up1 = -0.5*X.dot(inv(cov_mutual)).dot(X.transpose())
    X = row - mean_one
    up2 = -0.5*X.dot(inv(cov_mutual)).dot(X.transpose())
    ratio  = exp(up2[0,0] - up1[0,0])
    label = p_zero / (p_zero + (1-p_zero)*ratio)
    if label > 0.5:
        error += 1
        print(error/count)
'''


np.save("0_mean.npy", mean_zero)
np.save("1_mean.npy", mean_one)
np.save("mutual_cov.npy", cov_mutual)
