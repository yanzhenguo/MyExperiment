# -*- coding: utf-8 -*-

# 每个词赋予一个权重，用输入文本所有词的权重之和作逻辑回归，进行分类，每次迭代都去除低特征词。

import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector, SpatialDropout1D,Activation,Reshape
from keras import regularizers,constraints


num_words = 30000
max_len = 800
root_dir = '../../../'

def get_input():
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters=''
    tokenizer.fit_on_texts(texts[:25000])

    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=max_len)
    x_train = x[:25000]
    x_test = x[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    return x_train, y_train, x_test, y_test

def get_model():
    word_embed=0.05*np.random.rand(num_words,1)-0.05
    word_embed[0]=0
    #print(word_embed[:50])

    main_input = Input(shape=(max_len,))
    embedding1 = Embedding(num_words, 1,
                           weights=[word_embed],
                           )(main_input)
    x=Reshape((max_len,))(embedding1)
    output=Dense(1,
            weights=[np.ones([max_len,1]),np.zeros([1])],
            trainable=False,
            activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_input()
    for i in range(10):
        model = get_model()
        model.fit([x_train], y_train, batch_size=32, epochs=5,shuffle=True,
                  validation_data=([x_test], y_test))
        embedding_weights=np.reshape(model.get_layer('embedding_1').get_weights()[0],[num_words])
        embedding_weights=np.abs(embedding_weights)
        keras.backend.clear_session()
        sorted_index=np.argsort(embedding_weights)
        igore_index=set()
        for i in range(1000):
            igore_index.add(sorted_index[i])
        for i in range(25000):
            for j in range(max_len):
                if x_train[i,j] in igore_index:
                    x_train[i, j]=0
                if x_test[i,j] in igore_index:
                    x_test[i,j]=0
