# -*- coding: utf-8 -*-
"""hw4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bx8dKTcOOYphngogipD3zzybLfwtsXHr
"""

#!pip install -q keras
#!pip install matplotlib
# %cd '/content/drive/My Drive/ML/ml2019spring-hw4/src'

import numpy as np
from keras.models import load_model
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import sys
	
from numpy.random import seed
seed(66)
from tensorflow import set_random_seed
set_random_seed(66)

def parse(filename):
  f =  open(filename, 'r')
  position = [0, 299, 5, 7, 3, 15, 4]
  x_train = np.zeros((7, 48, 48, 1))
  y_train = np.zeros((7,1))
  for i, line in enumerate(f):
    if i > 300:
      break
    for j,pos in enumerate(position):
      if i == pos+1:
        raw = line.strip().replace(',', ' ').split('\n')
        temp = ' '.join(raw)
        raw_data = np.array(temp.split()).astype('float').reshape(1, 48*48+1)
        raw_y = raw_data[ : , 0 ]
        raw_x = raw_data[ : , 1: ] / 256
        x_train[j,:,:,:] = raw_x.reshape((48,48,1))
        y_train[j] = raw_y[0]
        break
  return x_train, to_categorical(y_train)

prefix = sys.argv[2]
x_train, y_train = parse(sys.argv[1])
model = load_model('model.h5')
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Calculate Saliency Map
def get_Saliency(model, input_image, output_index = 0):
    # Define the function to compute the gradient
    input_tensors = [model.input]
    gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
    compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    # Execute the function to compute the gradient
    x_value = np.expand_dims(input_image, axis=0)
    gradients = compute_gradients([x_value])[0][0]

    return gradients

def plot_img(img, cm, filename):
  plt.imshow(img, cmap=cm)
  plt.grid('off')
  plt.axis('off')
  if filename != None:
    plt.savefig(filename)
  plt.show()

def add_mask(img, mask):
  masked = np.zeros_like(img)
  avg = np.average(mask)
  for i in range(48*48):
    if mask[i] > avg:
      masked[i] = img[i]
  return masked

for i in range(7):
  print("class ", i)
  img = x_train[i]
  mask = get_Saliency(model, img, i)

  plot_img(img.reshape((48, 48)), 'gray', filename = None)
  plot_img(mask.reshape((48, 48)), 'jet', prefix+"fig1_"+str(i)+".jpg")

# visualize layer
def plot_layer_output(imgs, filename):
    plot_x, plot_y = 8, 16
    print("plotting......")
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        plt.subplot(plot_x, plot_y, x * plot_y + y+1)
        plt.grid('off')
        plt.axis('off')
        plt.imshow(imgs[x * plot_y + y], cmap='gray')
    plt.savefig(filename)
    plt.show()

# visualize layer
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def vis_img_in_filter(img = np.array(x_train[3]).reshape((1, 48, 48, 1)).astype(np.float64), 
                      layer_name = 'conv2d_2'):
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        img_asc = np.array(np.random.random((1,48,48,1)))
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
        
    plot_layer_output(img_ascs, prefix+"fig2_1.jpg")
    

vis_img_in_filter()

#https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
from keras.models import Model

layer_name = 'conv2d_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
layer_output = intermediate_layer_model.predict(x_train[3:4,:,:,:])
imgs = np.swapaxes(layer_output,0,3).reshape((128,44,44))

plot_layer_output(imgs, prefix+"fig2_2.jpg")


# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

def my_predict(imgs):
  gray_imgs = imgs[:,:,:,0:1]
  return(model.predict(gray_imgs))

def my_slic(img):
  return slic(img, n_segments=50)

seed(66)
set_random_seed(66)

for i in range(7):
  explainer = lime_image.LimeImageExplainer()
  explanation = explainer.explain_instance(x_train[i].reshape((48,48)), classifier_fn=my_predict, top_labels=5, hide_color=0, num_samples=1000, segmentation_fn=my_slic, random_seed=66)

  temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=5, hide_rest=False)
  plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
  plt.grid('off')
  plt.axis('off')
  plt.savefig(prefix+"fig3_"+str(i)+".jpg")
  plt.show()