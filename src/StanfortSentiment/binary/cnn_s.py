# -*- coding: utf-8 -*-

# 只利用句子级别的样本进行训练

import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical

num_words=10000
maxlen = 50
batch_size = 32
embedding_dims = 200
nb_filter = 250
filter_length = 3
nb_epoch = 10
train_size = 6920
test_size = 1821
val_size = 872
temp_dir='../../../temp/StanfortSentiment/binary/preprocess'

def get_input():
    with open(temp_dir+"texts.pkl",'rb') as f:
        texts=pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts[:train_size])
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(sequences[0:train_size], maxlen=maxlen)
    x_test = pad_sequences(sequences[train_size:train_size+test_size], maxlen=maxlen)
    x_val = pad_sequences(sequences[train_size+test_size:train_size+test_size+val_size], maxlen=maxlen)

    y_train=np.load(temp_dir+'Ytrain.npy')
    y_test=np.load(temp_dir+"Ytest.npy")
    y_val=np.load(temp_dir+'Yval.npy')

    return x_train,y_train,x_test,y_test

def bulid_model():
    x_train,y_train,x_test,y_test=get_input()
    model = Sequential()
    model.add(Embedding(num_words,embedding_dims,input_length=maxlen))
    model.add(Convolution1D(filters=nb_filter,
                            kernel_size =filter_length,
                            padding='valid',
                            activation='relu'))
    model.add(GlobalMaxPooling1D(name='pool'))
    # model.add(Dense(100,activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              validation_data=(x_test, y_test))

