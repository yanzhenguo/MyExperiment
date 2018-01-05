# -*- coding: utf-8 -*-
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
import get_data

root_dir='../../../'

def build_bow_cnn(num_words, max_len, save_tofile=False):
    #bowNNGN 的实现
    [x_train, y_train, x_test, y_test] = get_data.data_bow_cnn(num_words, max_len)
    main_input = Input(shape=(max_len,))
    init_method = keras.initializers.Orthogonal()
    # init_method = keras.initializers.RandomUniform(minval=0.4, maxval=0.5)
    x = Embedding(num_words, 1000, embeddings_initializer=init_method)(main_input)
    # x = Embedding(num_words, 1000)(main_input)
    x = AveragePooling1D(pool_size=3, strides=1)(x)

    # x = util.Bias_layer()(x)
    # x = Activation('relu')(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid', trainable=True, use_bias=True)(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
    # model.summary()
    if save_tofile:
        model.save(root_dir+'temp/keras_code/model/bowCNN.model.h5')
        np.save(root_dir+'temp/keras_code/model/embedding2.npy', model.get_layer('embedding_1').get_weights()[0])


def build_fnn(num_words):
    [x_train, y_train, x_test, y_test] = get_data.data_bow(num_words)
    main_input = Input(shape=(num_words,))
    # x = Dense(500, activation='relu')(main_input)
    # x = Dropout(0.8)(x)
    x = Dense(1, activation='sigmoid', use_bias=True)(main_input)
    model = Model(inputs=[main_input], outputs=[x])
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit([x_train], y_train,
              batch_size=32,
              epochs=5,
              shuffle=True,
              validation_data=([x_test], y_test))


def build_bow_cnn2(num_words=30000, max_len=500):
    # 将word2vec词向量作为额外输入
    # [x_train, y_train, x_test, y_test, embedding_matrix] = get_data.data_cnn(num_words, max_len, )
    [x_train, y_train, x_test, y_test, embedding_matrix2, weights_dense, embedding_matrix] =\
        get_data.data_bow_cnn2(num_words, max_len,)
    main_input = Input(shape=(max_len,))
    # model.add(Embedding(35000,50,input_length=500))
    init_method = keras.initializers.Orthogonal()
    # embedding1 = Embedding(num_words, 1000)(main_input)
    embedding1 = Embedding(num_words, 1000, weights=[embedding_matrix2], trainable=True)(main_input)
    x = AveragePooling1D(pool_size=5, strides=1, padding='valid')(embedding1)
    # x = GlobalMaxPooling1D()(x)
    # x=GlobalAveragePooling1D()(x)
    # x = Activation('relu')(x)

    embedding2 = Embedding(num_words, 300, weights=[embedding_matrix], trainable=False)(main_input)
    y = AveragePooling1D(pool_size=5, strides=1, padding='valid')(embedding2)
    y = Conv1D(filters=1000, kernel_size=1, padding='valid', use_bias=False, strides=1)(y)
    # y = GlobalMaxPooling1D()(y)

    # embedding3=Embedding(num_words,50,input_length=max_len,embeddings_initializer='normal')(input)
    # p=GlobalAveragePooling1D()(embedding3)

    # z=keras.layers.concatenate([x,y])
    z = keras.layers.add([x, y])
    # z = util.Bias_layer()(z)
    # z = keras.layers.average([x, y])
    z = Activation('relu')(z)
    z = GlobalMaxPooling1D()(z)
    # z=keras.layers.concatenate([x,y,p])
    # x=Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid', weights=weights_dense)(z)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))


def build_bow_wor2vec(num_words=30000, max_len=500):
    [x_train, y_train, x_test, y_test] = get_data.data_bow_cnn(num_words, max_len)
    embedding_matrix = util.get_word2vec_matrix(num_words=num_words)

    main_input = Input(shape=(max_len,))
    init_method = keras.initializers.Orthogonal()
    embedding1 = Embedding(num_words, 300, weights=[embedding_matrix], trainable=True)(main_input)
    x = AveragePooling1D(pool_size=3, strides=1)(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))


def build_cnn(num_words=30000, max_len=500):
    [x_train, y_train, x_test, y_test, embedding_matrix] = get_data.data_cnn(num_words, max_len)
    model = Sequential()
    model.add(Embedding(30000, 300, weights=[
        embedding_matrix], input_length=500, dropout=0.2, trainable=False))
    model.add(Conv1D(filters=1000, kernel_size=3,
                     padding='valid', activation='relu'))
    model.add(MaxPooling1D(name='pool'))
    model.add(Conv1D(filters=1000, kernel_size=3,
                     padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=10,
              validation_data=(x_test, y_test))


def build_lstm(num_words=30000, max_len=500):
    # 用lstm进行文本分类
    [x_train, y_train, x_test, y_test, embedding_matrix] = get_data.data_cnn(num_words, max_len)
    model = Sequential()
    model.add(Embedding(30000, 300, weights=[embedding_matrix], input_shape=(max_len,), trainable=False))
    model.add(LSTM(300))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=40,
              validation_data=(x_test, y_test))


def build_covlstm(num_words=30000, max_len=500):
    # 用convolutin+lstm进行文本分类
    [x_train, y_train, x_test, y_test, embedding_matrix] = get_data.data_cnn(num_words, max_len)
    model = Sequential()
    model.add(Embedding(num_words, 300, weights=[embedding_matrix], input_length=500, dropout=0.2, trainable=False))
    model.add(Conv1D(nb_filter=250, filter_length=3, border_mode='valid', activation='relu', subsample_length=1,
                            name='layer2'))
    model.add(MaxPooling1D(2, name='pool'))
    # model.add(LSTM(250,input_dim=250,input_length=498))
    model.add(LSTM(250))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_data=(x_test, y_test))

def build_lstmcov(num_words=30000, max_len=500):
    # 用lstm+convolutin进行文本分类
    [x_train, y_train, x_test, y_test, embedding_matrix] = get_data.data_cnn(num_words, max_len)
    model = Sequential()
    model.add(Embedding(101756, 300, weights=[embedding_matrix], input_length=max_len, dropout=0.2,
                        trainable=False))
    model.add(LSTM(300, return_sequences=True))
    model.add(Conv1D(filters=250, kernel_size=3, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_data=(x_test, y_test))


def build_bow_cnn_2(num_words=30000, max_len=500):
    # 将word2vec词向量作为额外输入
    # [x_train, y_train, x_test, y_test, embedding_matrix] = get_data.data_cnn(num_words, max_len, )
    [x_train, y_train, x_test, y_test, embedding_matrix2, weights_dense, embedding_matrix] = \
        get_data.data_bow_cnn2(num_words, max_len, )
    main_input = Input(shape=(max_len,))
    # model.add(Embedding(35000,50,input_length=500))
    init_method = keras.initializers.Orthogonal()
    # embedding1 = Embedding(num_words, 1000)(main_input)
    embedding1 = Embedding(num_words, 1000, weights=[embedding_matrix2], trainable=True)(main_input)
    x = AveragePooling1D(pool_size=4, strides=1, padding='valid')(embedding1)
    # x = GlobalMaxPooling1D()(x)
    # x=GlobalAveragePooling1D()(x)
    # x = Activation('relu')(x)

    embedding2 = Embedding(num_words, 1000, weights=[embedding_matrix2], trainable=False)(main_input)
    y = AveragePooling1D(pool_size=4, strides=1, padding='valid')(embedding2)
    y = Conv1D(filters=1000, kernel_size=1, padding='valid', use_bias=False, strides=1)(y)
    # y = GlobalMaxPooling1D()(y)

    # embedding3=Embedding(num_words,50,input_length=max_len,embeddings_initializer='normal')(input)
    # p=GlobalAveragePooling1D()(embedding3)

    # z=keras.layers.concatenate([x,y])
    z = keras.layers.add([x, y])
    # z = util.Bias_layer()(z)
    # z = keras.layers.average([x, y])
    z = Activation('relu')(z)
    z = GlobalMaxPooling1D()(z)
    # z=keras.layers.concatenate([x,y,p])
    # x=Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid', weights=weights_dense)(z)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))


if __name__ == '__main__':
    # build_bow_cnn(num_words=30000, max_len=500, save_tofile=False)
    build_fnn(num_words=5000)
    # build_bow_cnn2(num_words=30000, max_len=500)
    # build_bow_wor2vec(num_words=30000, max_len=500)
    # build_cnn()
    # build_lstm(max_len=300)
    # build_bow_cnn_2(num_words=30000, max_len=500)
