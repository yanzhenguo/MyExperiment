# -*- coding: utf-8 -*-
import pickle
import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, \
    BatchNormalization, RepeatVector, SpatialDropout1D, Lambda, Activation, Reshape, Conv1D

num_words = 30000
max_len = 1000
word_dimension = 1000
root_dir = '../../../'


def get_input():
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(texts[:25000])
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

    x = pad_sequences(sequences, maxlen=max_len)
    x_train = x[:25000]
    x_test = x[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    return x_train, y_train, x_test, y_test


def get_model():
    weight = np.ones((word_dimension, 1), dtype=np.float)
    weight[int(word_dimension / 2):] = -1 * np.ones([int(word_dimension / 2), 1], dtype=np.float)

    main_input = Input(shape=(max_len,))
    init_method = keras.initializers.Orthogonal()
    # embedding1 = Embedding(num_words, word_dimension,embeddings_initializer=init_method)(main_input)
    embedding1 = Embedding(num_words, word_dimension)(main_input)
    # x=SpatialDropout1D(0.3)(embedding1)
    # x=Dropout(rate=0.3)(embedding1)
    x = Conv1D(filters=250,kernel_size=3)(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_input()
    model = get_model()
    model.fit([x_train], y_train, batch_size=64, epochs=10, shuffle=True,
              validation_data=([x_test], y_test))

