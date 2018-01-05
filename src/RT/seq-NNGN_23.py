import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D, Concatenate

num_words = 15000
max_len = 70
word_dimension=1000
num_data=10662

root_dir='../../'

def get_input():
    with open(root_dir + "temp/RT/util/text.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_pad = pad_sequences(sequences, maxlen=max_len)

    x1=np.zeros((num_data, (max_len-1)*2), dtype=np.int)
    for i in range(num_data):
        for j in range(max_len-1):
            x1[i, j*2] = sequences_pad[i, j]
            x1[i, j*2+1] = sequences_pad[i][j+1]+num_words

    x2 = np.zeros((num_data, (max_len-2)*3), dtype=np.int)
    for i in range(num_data):
        for j in range(max_len-2):
            x2[i, j*3] = sequences_pad[i, j]
            x2[i, j*3+1] = sequences_pad[i][j+1]+num_words
            x2[i, j*3+2] = sequences_pad[i][j+2]+num_words*2

    y = np.zeros((num_data,), dtype=np.float32)
    y[5331:] = np.ones((5331,), dtype=np.float32)

    indice = np.arange(num_data)
    np.random.shuffle(indice)
    x1 = x1[indice]
    x2 = x2[indice]
    y = y[indice]

    return x1,x2,y


def get_model():
    print('begin to build model ...')
    input_1=Input(shape=((max_len-1)*2,))
    embedding1 = Embedding(num_words * 2, word_dimension, embeddings_initializer='normal')(input_1)
    x = AveragePooling1D(pool_size=2, strides=2)(embedding1)
    x = GlobalMaxPooling1D()(x)

    input_2 = Input(shape=((max_len - 2)*3, ))
    embedding2 = Embedding(num_words*3, word_dimension, embeddings_initializer='normal')(input_2)
    y = AveragePooling1D(pool_size=3,strides=3)(embedding2)
    y = GlobalMaxPooling1D()(y)

    z=Concatenate()([x,y])
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[input_1,input_2], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__=='__main__':
    x1,x2, y=get_input()
    acc = []
    for i in range(10):
        x1 = np.concatenate((x1[-1066:], x1[:-1066]), axis=0)
        x2 = np.concatenate((x2[-1066:], x2[:-1066]), axis=0)
        y = np.concatenate((y[-1066:], y[:-1066]), axis=0)
        model = get_model()
        hist = model.fit([x1,x2], y, batch_size=32, epochs=10,
                         validation_split=0.1)
        acc.append(max(hist.history['val_acc']))
        keras.backend.clear_session()
    print(sum(acc) / 10)
