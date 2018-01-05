# -*- coding: utf-8 -*-
"""use multi convolutional layer to capture features of different size of context"""
import codecs
import pickle
import keras
import sys
import numpy as np
from keras.layers import Dense, GlobalMaxPooling1D, Input, Embedding, \
    AveragePooling1D, GlobalAveragePooling1D, Activation, Conv1D, Dropout, MaxPooling1D, LSTM, Flatten, Concatenate
from keras.models import Model, load_model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import util
import os
import get_data

num_words=30000
word_embedding=300
max_len=400
datadir='../../../temp/imdb/keras_code/multi_cnn/data.npz'
def getdata():
    if os.path.exists(datadir):
        dic=np.load(datadir)
        Xtrain=dic['arr_0']
        Ytrain = dic['arr_1']
        Xtest = dic['arr_2']
        Ytest = dic['arr_3']
        embedding_matrix = dic['arr_4']
    else:
        [Xtrain, Ytrain, Xtest, Ytest, embedding_matrix] = get_data.data_cnn(num_words, max_len)
        np.savez(datadir,Xtrain, Ytrain, Xtest, Ytest, embedding_matrix)

    return Xtrain, Ytrain, Xtest, Ytest, embedding_matrix
Xtrain, Ytrain, Xtest, Ytest, embedding_matrix = getdata()
input = Input(shape=(max_len,))
x=Embedding(num_words, word_embedding, weights=[embedding_matrix], input_length=max_len, trainable=True)(input)
# model.add(Conv1D(filters=300, kernel_size=3,
#                  padding='valid', activation='relu'))
# model.add(MaxPooling1D(name='pool'))

cov1=Conv1D(filters=300, kernel_size=2,padding='same', activation='relu')(x)
# max1=GlobalMaxPooling1D()(cov1)

# cov2=MaxPooling1D()(cov1)
cov2=Conv1D(filters=300, kernel_size=2,padding='same', activation='relu')(cov1)
# max2=GlobalMaxPooling1D()(cov2)

y=keras.layers.concatenate([cov1,cov2],axis=1)
y=GlobalAveragePooling1D()(y)

y=Dense(250, activation='relu')(y)
# model.add(Dropout(0.2))
output=Dense(1, activation='sigmoid')(y)
model =Model(inputs=input,outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, batch_size=32, epochs=10,
          validation_data=(Xtest, Ytest))