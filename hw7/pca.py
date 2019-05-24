import os
import sys
import numpy as np 
from skimage.io import imread, imsave

IMAGE_PATH = sys.argv[1]

# Images for compression & reconstruction
test_image = sys.argv[2]
recon_image = sys.argv[3]

# Number of principal components used
k = 5

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

filelist = os.listdir(IMAGE_PATH) 
# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[1])).shape 

img_data = []
for filename in filelist:
    if filename.startswith("."):
        continue
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    img_data.append(tmp.flatten())
	
training_data = np.array(img_data).astype('float32')
# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 

# Use SVD to find the eigenvectors 
print("start svd")
u, s, v = np.linalg.svd(training_data.transpose(), full_matrices = False)  

# Load image & Normalize
picked_img = imread(os.path.join(IMAGE_PATH, test_image))  
X = picked_img.flatten().astype('float32') 
X -= mean
#print(X.shape)
    
# Compression
weight = np.array([X.transpose().dot(u[:,i]) for i in range(k)])  
# Reconstruction
reconstruct = process(u[:,:5].dot(weight) + mean)
imsave(recon_image, reconstruct.reshape(img_shape))