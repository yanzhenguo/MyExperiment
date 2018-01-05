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
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D
from tensorflow.python.lib.io.file_io import FileIO

num_words = 30000
max_len = 1000
embedding_dimension = 2100
FLAGS = None
def main():
    f = FileIO(os.path.join(FLAGS.buckets, "texts.pkl"), mode='r+')
    texts = pickle.load(f)
    f.close()

    tokenizer = Tokenizer(nb_words=num_words)
    tokenizer.fit_on_texts(texts[0:25000])
    # sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    sequences = []
    for i in range(50000):
        t = []
        tokens = texts[i].lower().split(' ')
        for j in range(len(tokens)):
            index = word_index.get(tokens[j], 0)
            if index < num_words:
                t.append(index)
            else:
                t.append(0)
        sequences.append(t)

    print('Found %s unique tokens.' % len(word_index))

    data1 = pad_sequences(sequences[0:25000], maxlen=max_len)
    data2 = pad_sequences(sequences[25000:50000], maxlen=max_len)
    Ytrain = np.zeros((25000,), dtype=np.float32)
    Ytest = np.zeros((25000,), dtype=np.float32)
    Ytrain[12500:25000] = np.ones((12500,), dtype=np.float32)
    Ytest[12500:25000] = np.ones((12500,), dtype=np.float32)

    Xtrain = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
    Xtest = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
    for i in range(25000):
        for j in range(max_len - 2):
            Xtrain[i, j * 3] = data1[i, j]
            Xtrain[i, j * 3 + 1] = data1[i][j + 1] + num_words
            Xtrain[i, j * 3 + 2] = data1[i][j + 2] + num_words * 2
    for i in range(25000):
        for j in range(max_len - 3):
            Xtest[i, j * 3] = data2[i, j]
            Xtest[i, j * 3 + 1] = data2[i][j + 1] + num_words
            Xtest[i, j * 3 + 2] = data2[i][j + 2] + num_words * 2

    indice1 = np.arange(25000)
    np.random.shuffle(indice1)
    Xtrain = Xtrain[indice1]
    Ytrain = Ytrain[indice1]

    indice2 = np.arange(25000)
    np.random.shuffle(indice2)
    Xtest = Xtest[indice2]
    Ytest = Ytest[indice2]
    print('begin to build model ...')
    main_input = Input(shape=((max_len - 2) * 3,))
    embedding1 = Embedding(num_words * 3, embedding_dimension, init='orthogonal')(
        main_input)
    x = AveragePooling1D(pool_length=3)(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(input=main_input, output=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit([Xtrain], Ytrain, batch_size=32, nb_epoch=10, verbose=2, validation_data=([Xtest], Ytest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    # 获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    main()
