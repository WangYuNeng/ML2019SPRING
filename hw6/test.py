
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import load_model
import numpy as np
import pandas as pd
import csv
import jieba
import pickle
import sys

test_x_path = sys.argv[1]
dictionary = sys.argv[2]

model_list = ["model.h5"]

output_path = sys.argv[3]

MAX_SEQUENCE_LENGTH = 100

jieba.set_dictionary(dictionary)

# testing data

raw_test_x = list()
df = pd.read_csv(test_x_path)
for sentence in df["comment"]:
  seg_list = jieba.cut(sentence.replace(" ", ""), cut_all=False)
  raw_test_x.append( " ".join(seg_list) )

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
sequences = tokenizer.texts_to_sequences(raw_test_x) 
test_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    def get_config(self):
        base_config = super(Attention, self).get_config()
        base_config["attention_size"] = self.attention_size
        return base_config

test_y = np.zeros((20000,2))

for modelname in model_list:
  model = load_model(modelname, custom_objects={'Attention': Attention})
  pred = model.predict(test_data, batch_size=64)
  test_y += pred
  del model



with open(output_path, "w") as f:
  f.write("id,label\n")
  i = 0
  for array in test_y:
    label = 0;
    if array[0] < array[1]:
      label = 1
    line = str(i) + "," + str(label) + '\n'
    f.write(line)
    i += 1