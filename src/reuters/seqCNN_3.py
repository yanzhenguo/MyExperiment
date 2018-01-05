import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D
num_words = 30000
max_len = 500



def get_input():
    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(path='reuters.npz',
                                                                            num_words=num_words,
                                                                            test_split=0.1, )

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    data1 = pad_sequences(x_train, maxlen=max_len)
    data2 = pad_sequences(x_test, maxlen=max_len)

    Xtrain = np.zeros((len(x_train), (max_len-2)*3), dtype=np.int)
    Xtest = np.zeros((len(x_test), (max_len-2)*3), dtype=np.int)
    for i in range(len(x_train)):
        for j in range(max_len-2):
            Xtrain[i, j*3] = data1[i, j]
            Xtrain[i, j*3+1] = data1[i][j+1]+num_words
            Xtrain[i, j*3+2] = data1[i][j+2]+num_words*2
    for i in range(len(x_test)):
        for j in range(max_len-2):
            Xtest[i, j*3] = data2[i, j]
            Xtest[i, j*3+1] = data2[i][j+1]+num_words
            Xtest[i, j*3+2] = data2[i][j+2]+num_words*2


    return Xtrain,y_train,Xtest,y_test


def get_model():
    print('begin to build model ...')
    main_input = Input(shape=((max_len - 2)*3, ))
    embedding1 = Embedding(num_words*3, 800, input_length=(max_len-2)*3, embeddings_initializer='normal')(main_input)
    x = AveragePooling1D(pool_size=3,strides=3)(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(46, activation='softmax')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__=='__main__':
    x_train, y_train, x_test, y_test=get_input()
    model=get_model()
    model.fit(x_train, y_train, batch_size=32, epochs=20, shuffle=True, validation_data=(x_test, y_test))
