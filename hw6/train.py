
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, BatchNormalization
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras import callbacks
from keras.models import load_model
from keras.utils import Sequence
import numpy as np
from gensim.models import word2vec

import random
import os
import sys
import pandas as pd
import csv
import jieba
jieba.set_dictionary(sys.argv[4])
wordlist_path = "x_word.txt"

x_word = open(wordlist_path, "wb")

# training data
df = pd.read_csv(sys.argv[1])
for sentence in df["comment"]:
  seg_list = jieba.cut(sentence.replace(" ", ""), cut_all=False)
  x_word.write( " ".join(seg_list).encode('utf-8') )
  x_word.write(b'\n')

  
# testing data
df = pd.read_csv(sys.argv[3])
for sentence in df["comment"]:
  seg_list = jieba.cut(sentence.replace(" ", ""), cut_all=False)
  x_word.write( " ".join(seg_list).encode('utf-8') )
  x_word.write(b'\n')
  
x_word.close()

x_word = list()
f = open(wordlist_path, "r")
for line in f:
  x_word.append(line[:-1])

MAX_SEQUENCE_LENGTH = 100



tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(x_word)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

count_thres = 3
low_count_words = [w for w,c in tokenizer.word_counts.items() if c < count_thres]
for w in low_count_words:
    del tokenizer.word_index[w]
    del tokenizer.word_docs[w]
    del tokenizer.word_counts[w]
sequences = tokenizer.texts_to_sequences(x_word) 
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) 
print('Shape of data tensor:', data.shape)

yf = pd.read_csv(sys.argv[2])
raw_y = yf["label"]

  
train_y = to_categorical(raw_y)

MAX_NB_WORDS = len(word_index) + 1
EMBEDDING_DIM = 300

wordlist = word2vec.LineSentence(wordlist_path)

model = word2vec.Word2Vec(
        wordlist,
        size=EMBEDDING_DIM,
        window=3,
        min_count=3,
        workers=8)
model.train(wordlist, total_examples=len(x_word), epochs=30)

weight_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
vocab = tokenizer.word_index


for word, i in vocab.items():
  try:
    weight_matrix[i] = model.wv[word]
  except KeyError:
    np.random.seed(66) # fixed seed for unknown
    weight_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
del model


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


def get_model(seed):
  random.seed(seed)
  lr = random.uniform(0.0001, 0.00005)
  CNN = random.choice([True, False])
  Conv_size = random.choice([64, 128, 256, 512])
  LSTM_size = random.choice([128, 256, 512, 1024])
  Dense_size = random.choice([128, 256, 512, 1024])
  drop1 = random.uniform(0.1, 0.3)
  drop2 = random.uniform(0.1, 0.3)  
  drop3 = random.uniform(0.2, 0.5)
  recur_drop = random.uniform(0.1, 0.3)
  
  
  adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

  model = Sequential()
  model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[weight_matrix], input_length=data.shape[1], trainable=True))
  if CNN == True:
    model.add(Convolution1D(Conv_size, 3, padding='same', strides = 1))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
  model.add(Dropout(drop1))
  model.add(Bidirectional(LSTM(LSTM_size, dropout=drop2, recurrent_dropout=recur_drop, return_sequences=True)))
  model.add(Attention(100))
  model.add(Dense(Dense_size, activation='relu'))
  model.add(Dropout(drop3))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model



epochs = 20
batch_size = 64

model = get_model(11)
filepath = "models/{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5"
earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
history = model.fit(data[:119018,:], train_y[:119018,:], epochs=epochs, batch_size=batch_size,validation_split=0.1, callbacks=[earlystop])
