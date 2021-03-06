# -*- coding: utf-8 -*-
import codecs
import pickle
import os
import numpy as np
import keras
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D, SpatialDropout1D, GRU
from tensorflow.python.lib.io.file_io import FileIO

num_words = 30000
max_len = 51
num_data = 10662
num_train = 9596
embedding_dimension = 300
root_dir = '../../'


def get_input():
    with open(root_dir + "temp/RT/util/text.pkl", 'rb') as f:
        texts = pickle.load(f)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_reverse = [list(reversed(seq)) for seq in sequences]

    x = pad_sequences(sequences, maxlen=max_len)
    x_reverse = pad_sequences(sequences_reverse, maxlen=max_len)

    word_index = tokenizer.word_index
    embeddings_index = {}
    wordX = np.load(open("/home/yan/my_datasets/glove/embedding.300d.npy", mode='rb'))
    allwords = pickle.load(open("/home/yan/my_datasets/glove/words.pkl", mode='rb'))
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    y = np.zeros((num_data,), dtype=np.float32)
    y[5331:] = np.ones((5331,), dtype=np.float32)

    result = []
    indice = np.arange(num_data)
    np.random.shuffle(indice)
    result.append(x[indice])
    result.append(x_reverse[indice])
    result.append(y[indice])

    result.append(embedding_matrix)
    return result


def get_model(embedding_matrix):
    input_1 = Input((max_len,))
    embedding_1 = Embedding(num_words, 300,
                            weights=[embedding_matrix],
                            trainable=False)(input_1)
    # x = SpatialDropout1D(0.25)(embedding_1)
    x = GRU(300,
            dropout=0.2,
            recurrent_dropout=0.2,
            activation='relu')(embedding_1)

    input_2 = Input((max_len,))
    embedding_2 = Embedding(num_words, 300,
                            weights=[embedding_matrix],
                            trainable=False)(input_2)
    # y = SpatialDropout1D(0.25)(embedding_2)
    y = GRU(300,
            dropout=0.2,
            recurrent_dropout=0.2,
            activation='relu')(embedding_2)

    embedding_3= Embedding(num_words,500)(input_1)
    z=AveragePooling1D(pool_size=3,strides=1)(embedding_3)
    z=GlobalMaxPooling1D()(z)

    a = keras.layers.concatenate([x, y,z])
    output_1 = Dense(1, activation='sigmoid')(a)

    model = Model(inputs=[input_1, input_2], outputs=[output_1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x, x_reverse, y, embedding_matrix = get_input()
    acc = []
    for i in range(10):
        x = np.concatenate((x[-1066:], x[:-1066]), axis=0)
        x_reverse = np.concatenate((x_reverse[-1066:], x_reverse[:-1066]), axis=0)
        y = np.concatenate((y[-1066:], y[:-1066]), axis=0)
        model = get_model(embedding_matrix)
        hist = model.fit([x, x_reverse], y, batch_size=32, epochs=10,
                         validation_split=0.1)
        acc.append(max(hist.history['val_acc']))
        keras.backend.clear_session()
    print(sum(acc) / 10)
