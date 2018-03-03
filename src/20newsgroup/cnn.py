# 使用cnn进行分类，一个卷积层加一个全局最大池化层
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D
# from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical

num_words = 30000
max_len = 1000
word_dimension = 300
num_train = 11314
num_test = 7532
data_dir = '../../temp/20newsgroup/util/'


def get_data():
    with open(data_dir + 'texts.pkl', 'rb') as f:
        texts = pickle.load(f)

    y_train = np.load(data_dir + 'Ytrain.npy')
    y_test = np.load(data_dir + 'Ytest.npy')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts[:num_train])

    sequence1 = tokenizer.texts_to_sequences(texts[:num_train])
    sequence2 = tokenizer.texts_to_sequences(texts[num_train:])
    word_index = tokenizer.word_index

    # sequences=[]
    # for i in range(50000):
    #     t=[]
    #     tokens=texts[i].lower().split(' ')
    #     for j in range(len(tokens)):
    #         index=word_index.get(tokens[j],0)
    #         if index<num_words:
    #             t.append(index)
    #         else:
    #             t.append(0)
    #     sequences.append(t)

    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(sequence1, maxlen=max_len)
    x_test = pad_sequences(sequence2, maxlen=max_len)

    return x_train, y_train, x_test, y_test


def build_model():
    x_train, y_train, x_test, y_test = get_data()
    model = Sequential()
    # model.add(Embedding(35000,50,input_length=500))
    model.add(Embedding(num_words, word_dimension, input_length=max_len))
    model.add(Conv1D(filters=250, kernel_size=3, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              shuffle=True,
              validation_data=(x_test, y_test))


if __name__ == '__main__':
    build_model()
