import numpy as np
import sys
from keras.models import Model
import keras.layers as layers


# keras implement
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    
    channel_axis = -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)
    return layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
   
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
    
    return layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              classes=7):
    
    # Determine proper input shape and default size.
    rows = input_shape[0]
    cols = input_shape[1]
    default_size = rows

    img_input = layers.Input(shape=input_shape)

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = layers.Dropout(0.1, name='dropout128')(x)
    
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=6)
    

    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(7, activation="softmax")(x)
    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    return model

def create_model(input_shape,
              alpha,
              depth_multiplier,
              classes=7):
    model = MobileNet(input_shape=input_shape,
              alpha=alpha,
              depth_multiplier=depth_multiplier,
              classes=classes)
    print(model.summary())
    return model