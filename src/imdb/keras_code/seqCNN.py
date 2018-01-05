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
max_len = 500
root_dir = '../../../'


with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
    texts = pickle.load(f)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[:25000])
word_index = tokenizer.word_index


# def getnewSeq():
#     sequences = []
#     for i in range(50000):
#         t = []
#         tokens = texts[i].lower().split(' ')
#         for j in range(len(tokens)):
#             index = word_index.get(tokens[j], 0)
#             if index < num_words:
#                 t.append(index)
#             else:
#                 t.append(0)
#         sequences.append(t)
#     return sequences
#
#
# def getInitialInput(sequences):
#     xtrain = pad_sequences(sequences[0:25000], maxlen=max_len)
#     xtest = pad_sequences(sequences[25000:50000], maxlen=max_len)
#     ytrain = np.zeros((25000,), dtype=np.float32)
#     ytest = np.zeros((25000,), dtype=np.float32)
#     ytrain[12500:25000] = np.ones((12500,), dtype=np.float32)
#     ytest[12500:25000] = np.ones((12500,), dtype=np.float32)
#     return xtrain, ytrain, xtest, ytest
#
#
# def shuffleData(data):
#     indice1 = np.arange(len(data[0]))
#     np.random.shuffle(indice1)
#     for i in range(len(data)):
#         data[i] = data[i][indice1]
#     return data


def getinput_1():
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    Xtrain, Ytrain, Xtest, Ytest = imdb.get_input()
    util.shuffleData([Xtrain, Ytrain, Xtest, Ytest])
    return [Xtrain, Ytrain, Xtest, Ytest]


def getinput_2():
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    Xtrain, Ytrain, Xtest, Ytest = imdb.get_input()
    Xtr = np.zeros((25000, (max_len - 1) * 2), dtype=np.int)
    Xte = np.zeros((25000, (max_len - 1) * 2), dtype=np.int)
    for i in range(25000):
        for j in range(max_len - 1):
            Xtr[i, j * 2] = Xtrain[i, j]
            Xtr[i, j * 2 + 1] = Xtrain[i][j + 1] + num_words
    for i in range(25000):
        for j in range(max_len - 1):
            Xte[i, j * 2] = Xtest[i, j]
            Xte[i, j * 2 + 1] = Xtest[i][j + 1] + num_words
    Xtrain = Xtr
    Xtest = Xte
    util.shuffleData([Xtrain, Ytrain, Xtest, Ytest])
    return [Xtrain, Ytrain, Xtest, Ytest]


def getinput_3(shuffle=False):
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    data1, Ytrain, data2, Ytest = imdb.get_input()
    Xtrain = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
    Xtest = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
    for i in range(25000):
        for j in range(max_len - 2):
            Xtrain[i, j * 3] = data1[i, j]
            Xtrain[i, j * 3 + 1] = data1[i][j + 1] + num_words
            Xtrain[i, j * 3 + 2] = data1[i][j + 2] + num_words * 2
    for i in range(25000):
        for j in range(max_len - 2):
            Xtest[i, j * 3] = data2[i, j]
            Xtest[i, j * 3 + 1] = data2[i][j + 1] + num_words
            Xtest[i, j * 3 + 2] = data2[i][j + 2] + num_words * 2
    if shuffle:
        util.shuffleData([Xtrain, Ytrain, Xtest, Ytest])
    return [Xtrain, Ytrain, Xtest, Ytest]


def getinput_4():
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    data1, Ytrain, data2, Ytest = imdb.get_input()

    Xtrain = np.zeros((25000, (max_len - 3) * 4), dtype=np.int)
    Xtest = np.zeros((25000, (max_len - 3) * 4), dtype=np.int)
    for i in range(25000):
        for j in range(max_len - 3):
            Xtrain[i, j * 4] = data1[i, j]
            Xtrain[i, j * 4 + 1] = data1[i][j + 1] + num_words
            Xtrain[i, j * 4 + 2] = data1[i][j + 2] + num_words * 2
            Xtrain[i, j * 4 + 3] = data1[i][j + 3] + num_words * 3
    for i in range(25000):
        for j in range(max_len - 3):
            Xtest[i, j * 4] = data2[i, j]
            Xtest[i, j * 4 + 1] = data2[i][j + 1] + num_words
            Xtest[i, j * 4 + 2] = data2[i][j + 2] + num_words * 2
            Xtest[i, j * 4 + 3] = data2[i][j + 3] + num_words * 3
    util.shuffleData([Xtrain, Ytrain, Xtest, Ytest])
    return [Xtrain, Ytrain, Xtest, Ytest]


def buildModel_1():
    input_data = getinput_1()
    input = Input(shape=(max_len,))
    embedding1 = Embedding(num_words, 1000, input_length=max_len, embeddings_initializer='normal')(input)
    # x=AveragePooling1D(pool_size=2)(embedding1)
    # x=GlobalMaxPooling1D()(x)
    x = GlobalAveragePooling1D()(embedding1)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit([input_data[0]], input_data[1], batch_size=32, epochs=50,
              validation_data=([input_data[2]], input_data[3]))


def buildModel_2():
    input_data = getinput_2()
    input = Input(shape=((max_len - 1) * 2,))
    embedding1 = Embedding(num_words * 2, 1000, input_length=(max_len - 1) * 2, embeddings_initializer='normal')(input)
    x = AveragePooling1D(pool_size=2)(embedding1)
    x = GlobalMaxPooling1D()(x)
    # x=LSTM(500)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit([input_data[0]], input_data[1], batch_size=32, epochs=50, validation_data=([input_data[2]], input_data[3]))


def buildModel_3():
    input_data = getinput_3()
    main_input = Input(shape=((max_len - 2) * 3,))
    embedding1 = Embedding(num_words * 3, 500, embeddings_initializer=keras.initializers.Orthogonal())(main_input)
    # embedding1 = Embedding(num_words * 3, 500)(main_input)
    x = AveragePooling1D(pool_size=3)(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit([input_data[0]], input_data[1], batch_size=32, epochs=12, validation_data=([input_data[2]], input_data[3]))
    #model.save('../temp/seqCNN3.h5')


def buildModel_4():
    input_data = getinput_4()
    main_input = Input(shape=((max_len - 3) * 4,))
    embedding1 = Embedding(num_words * 4, 500, embeddings_initializer=keras.initializers.Orthogonal())(
        main_input)
    x = AveragePooling1D(pool_size=4)(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit([input_data[0]], input_data[1], batch_size=32, epochs=50,
              validation_data=([input_data[2]], input_data[3]))

def seq3_wv():
    model_seqcnn = load_model('../temp/seqCNN3.h5')
    weights_dense = model_seqcnn.get_layer('dense_1').get_weights();
    embedding_matrix1 = model_seqcnn.get_layer('embedding_1').get_weights();
    input_data = getinput_3()
    main_input = Input(shape=((max_len - 2) * 3,))
    # embedding1 = Embedding(num_words * 3, 500, embeddings_initializer=keras.initializers.Orthogonal())(main_input)
    embedding1 = Embedding(num_words * 3, 500, weights=embedding_matrix1)(main_input)
    x = AveragePooling1D(pool_size=3)(embedding1)

    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    embedding_matrix = imdb.get_word2vec_matrix()
    xtrain, ytrain, xtest, ytest = imdb.get_input()
    input2 = Input(shape=(max_len,))
    embedding2 = Embedding(num_words, 300, weights=[embedding_matrix], trainable=False)(input2)
    y = AveragePooling1D(pool_size=3, strides=1, padding='valid')(embedding2)
    y = Conv1D(filters=500, kernel_size=1, padding='valid', use_bias=False, strides=1)(y)

    z = keras.layers.add([x, y])
    # z = util.Bias_layer()(z)
    # z = keras.layers.average([x, y])
    z = Activation('relu')(z)
    z = GlobalMaxPooling1D()(z)
    # z=keras.layers.concatenate([x,y,p])
    # x=Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid', weights=weights_dense)(z)

    model = Model(inputs=[main_input, input2], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    indice1 = np.arange(25000)
    input_data[0] = input_data[0][indice1]
    xtrain = xtrain[indice1]
    ytrain = ytrain[indice1]
    input_data[2] = input_data[2][indice1]
    xtest = xtest[indice1]
    ytest = ytest[indice1]

    model.fit([input_data[0], xtrain], ytrain, batch_size=32, epochs=50, validation_data=([input_data[2], xtest], ytest))


if __name__ == '__main__':

    buildModel_3()
    # seq3_wv()
