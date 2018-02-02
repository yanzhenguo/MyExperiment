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
word_dim= 1000
ngram=4
FLAGS = None


def main():
    global ngram
    f = FileIO(os.path.join(FLAGS.buckets, "texts.pkl"), mode='r+')
    texts = pickle.load(f)
    f.close()

    tokenizer = Tokenizer(nb_words=num_words)
    tokenizer.filters=''
    tokenizer.fit_on_texts(texts[0:25000])
    sequences = tokenizer.texts_to_sequences(texts)
    # word_index = tokenizer.word_index
    # sequences = []
    # for i in range(50000):
    #     t = []
    #     tokens = texts[i].lower().split(' ')
    #     for j in range(len(tokens)):
    #         index = word_index.get(tokens[j], 0)
    #         if index < num_words:
    #             t.append(index)
    #         else:
    #             t.append(0)
    #     sequences.append(t)

    # print('Found %s unique tokens.' % len(word_index))

    data1 = pad_sequences(sequences[0:25000], maxlen=max_len)
    data2 = pad_sequences(sequences[25000:50000], maxlen=max_len)
    Ytrain = np.zeros((25000,), dtype=np.float32)
    Ytest = np.zeros((25000,), dtype=np.float32)
    Ytrain[12500:25000] = np.ones((12500,), dtype=np.float32)
    Ytest[12500:25000] = np.ones((12500,), dtype=np.float32)

    Xtrain = np.zeros((25000, (max_len - ngram + 1) * ngram), dtype=np.int)
    Xtest = np.zeros((25000, (max_len - ngram + 1) * ngram), dtype=np.int)

    id_range = np.arange(max_len - ngram + 1)
    for i in range(ngram):
        Xtrain[:, id_range * ngram + i] = data1[:, id_range + i] + num_words * i
        Xtest[:, id_range * ngram + i] = data2[:, id_range + i] + num_words * i

    print('begin to build model ...')
    main_input = Input(shape=((max_len - ngram + 1) * ngram,))
    # embedding1 = Embedding(num_words * ngram, word_dim, embeddings_initializer=keras.initializers.Orthogonal())(main_input)
    embedding1 = Embedding(num_words * ngram, word_dim)(main_input)
    x = AveragePooling1D(pool_size=ngram)(embedding1)
    x = GlobalMaxPooling1D()(x)

    weight = np.ones((word_dim, 1), dtype=np.float)
    weight[int(word_dim / 2):] = -1 * np.ones([int(word_dim / 2), 1], dtype=np.float)
    output = Dense(1,
                   weights=[weight, np.zeros([1])],
                   trainable=False,
                   activation='sigmoid')(x)

    model = Model(input=main_input, output=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit([Xtrain], Ytrain,
              batch_size=32,
              shuffle=True,
              nb_epoch=15,
              verbose=2,
              validation_data=([Xtest], Ytest))


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
