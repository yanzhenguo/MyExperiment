# -*- coding: utf-8 -*-
import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D
import util

num_words = 30000
max_len = 1000
word_dim=600
ngram=3
root_dir = '../../../'


def get_input(ngram=3):
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    data1, Ytrain, data2, Ytest = imdb.get_input()
    Xtrain = np.zeros((25000, (max_len - ngram+1) * ngram), dtype=np.int)
    Xtest = np.zeros((25000, (max_len - ngram+1) * ngram), dtype=np.int)

    id_range=np.arange(max_len-ngram+1)
    for i in range(ngram):
        Xtrain[:, id_range * ngram+i] = data1[:, id_range+i]+num_words*i
        Xtest[:, id_range * ngram + i] = data2[:, id_range + i] + num_words*i

    return [Xtrain, Ytrain, Xtest, Ytest]


def buld_model(ngram=3):
    input_data = get_input(ngram=ngram)
    main_input = Input(shape=((max_len - ngram+1) * ngram,))
    # embedding1 = Embedding(num_words * ngram, word_dim, embeddings_initializer=keras.initializers.Orthogonal())(main_input)
    embedding1 = Embedding(num_words * ngram, word_dim)(main_input)
    x = AveragePooling1D(pool_size=ngram)(embedding1)
    x = GlobalMaxPooling1D()(x)
    # output = Dense(1, activation='sigmoid')(x)

    weight = np.ones((word_dim, 1), dtype=np.float)
    weight[int(word_dim / 2):] = -1 * np.ones([int(word_dim / 2), 1], dtype=np.float)
    output = Dense(1,
                   weights=[weight, np.zeros([1])],
                   trainable=False,
                   activation='sigmoid')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit([input_data[0]], input_data[1],
              batch_size=32,
              shuffle=True,
              epochs=12,
              validation_data=([input_data[2]], input_data[3]))
    # model.save('../temp/seqCNN3.h5')


if __name__ == '__main__':
    buld_model(ngram=ngram)
    # seq3_wv()
