# -*- coding: utf-8 -*-

'''使用双向gru进行分类'''
import pickle
import os
import numpy as np
import argparse
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector,GRU, SpatialDropout1D
from tensorflow.python.lib.io.file_io import FileIO

num_words=30000
max_len=400
word_dimension=300
FLAGS = None


def get_input():
    with FileIO(os.path.join(FLAGS.buckets, "imdb/texts.pkl"), mode='r+') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts[0:25000])
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_reverse = [list(reversed(seq)) for seq in sequences]

    x = pad_sequences(sequences, maxlen=max_len)
    x_reverse = pad_sequences(sequences_reverse, maxlen=max_len)

    word_index = tokenizer.word_index
    embeddings_index = {}
    wordX = np.load(FileIO(os.path.join(FLAGS.buckets, "glove/embedding.300d.npy"), mode='r+'))
    allwords = pickle.load(FileIO(os.path.join(FLAGS.buckets, "glove/words.pkl"), mode='r+'))
    print(len(allwords))
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    x_train_0 = x[:25000]
    x_train_1=x_reverse[:25000]
    x_test_0 = x[25000:]
    x_test_1=x_reverse[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    return x_train_0,x_train_1, y_train, x_test_0, x_test_1, y_test, embedding_matrix

def get_model(embedding_matrix):
    # input_1 = Input((max_len,))
    # embedding_1 = Embedding(num_words, 300,
    #                         # weights=[embedding_matrix],
    #                         trainable=True)(input_1)
    # x = SpatialDropout1D(0.25)(embedding_1)
    # x = GRU(300,
    #         dropout=0.2,
    #         recurrent_dropout=0.2,
    #         activation='relu')(x)
    # # x = Dense(300, activation='relu')(x)
    #
    # input_2 = Input((max_len,))
    # embedding_2 = Embedding(num_words, 300,
    #                         # weights=[embedding_matrix],
    #                         trainable=True)(input_2)
    # y = SpatialDropout1D(0.25)(embedding_2)
    # y = GRU(300,
    #         dropout=0.2,
    #         recurrent_dropout=0.2,
    #         activation='relu')(y)
    # # y = Dense(300, activation='relu')(y)
    input_1 = Input((max_len,))
    init_1=keras.initializers.random_uniform(minval=-0.05,maxval=0.05)
    embedding = Embedding(num_words, 300,
                             weights=[embedding_matrix],
                          # embeddings_initializer=init_1,
                            trainable=False)
    embedding_1=embedding(input_1)
    # x = SpatialDropout1D(0.25)(embedding_1)
    x = GRU(300,
            dropout=0.2,
            recurrent_dropout=0.2,
            # activation='relu'
            )(embedding_1)
    # x = Dense(300, activation='relu')(x)

    input_2 = Input((max_len,))
    embedding_2 = embedding(input_2)
    # y = SpatialDropout1D(0.25)(embedding_2)
    y = GRU(300,
            dropout=0.2,
            recurrent_dropout=0.2,
            # activation='relu'
            )(embedding_2)
    # y = Dense(300, activation='relu')(y)

    a = keras.layers.concatenate([x, y])

    output_1=Dense(1,activation='sigmoid')(a)
    model=Model(inputs=[input_1,input_2],outputs=[output_1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    x_train_0, x_train_1, y_train, x_test_0, x_test_1, y_test, embedding_matrix=get_input()
    model=get_model(embedding_matrix)
    model.fit([x_train_0,x_train_1], y_train,
              batch_size=128,
              verbose=2,
              epochs=30,
              shuffle=True,
              validation_data=([x_test_0,x_test_1], [y_test]))