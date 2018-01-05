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

num_words = 15000
max_len = 70
word_dimension=600
num_data=10662

root_dir='../../'

def get_input():
    with open(root_dir + "temp/RT/util/text.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_pad = pad_sequences(sequences, maxlen=max_len)

    X = np.zeros((num_data, (max_len-2)*3), dtype=np.int)
    for i in range(num_data):
        for j in range(max_len-2):
            X[i, j*3] = sequences_pad[i, j]
            X[i, j*3+1] = sequences_pad[i][j+1]+num_words
            X[i, j*3+2] = sequences_pad[i][j+2]+num_words*2

    y = np.zeros((num_data,), dtype=np.float32)
    y[5331:] = np.ones((5331,), dtype=np.float32)

    indice = np.arange(num_data)
    np.random.shuffle(indice)
    x = X[indice]
    y = y[indice]

    return x,y


def get_model():
    print('begin to build model ...')
    main_input = Input(shape=((max_len - 2)*3, ))
    embedding1 = Embedding(num_words*3, word_dimension, embeddings_initializer='normal')(main_input)
    x = AveragePooling1D(pool_size=3,strides=3)(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__=='__main__':
    x, y=get_input()
    acc = []
    for i in range(10):
        x = np.concatenate((x[-1066:], x[:-1066]), axis=0)
        y = np.concatenate((y[-1066:], y[:-1066]), axis=0)
        model = get_model()
        hist = model.fit(x, y, batch_size=32, epochs=10,
                         validation_split=0.1)
        acc.append(max(hist.history['val_acc']))
        keras.backend.clear_session()
    print(sum(acc) / 10)
