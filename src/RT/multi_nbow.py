# -*- coding: utf-8 -*-
'''分别计算1-gram,2-gram,3-gram的向量平均值，然后使用多层全连接层进行处理，
得到三个文档表示向量，连接到一起组成新的文档向量，然后进行分类'''
import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector, SpatialDropout1D


num_words = 30000
num_gram2=100000
num_gram3=200000
max_len = 51
num_data=10662
num_train=9596
word_dimension = 10
root_dir = '../../'


def get_input():
    with open(root_dir + "temp/RT/util/text.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    text2=[]
    for seq in sequences:
        t=[]
        for i in range(len(seq)-1):
            t.append(str(seq[i])+'*'+str(seq[i+1]))
        text2.append(' '.join(t))
    tokenizer2=Tokenizer(num_words=num_gram2)
    tokenizer2.fit_on_texts(text2)
    sequences_2=tokenizer2.texts_to_sequences(text2)

    text3=[]
    for seq in sequences:
        t=[]
        for i in range(len(seq)-2):
            t.append(str(seq[i])+'*'+str(seq[i+1])+'*'+str(seq[i+2]))
        text3.append(' '.join(t))
    tokenizer3=Tokenizer(num_words=num_gram3)
    tokenizer3.fit_on_texts(text3)
    sequences_3=tokenizer3.texts_to_sequences(text2)

    x_1 = pad_sequences(sequences, maxlen=max_len)
    x_2 = pad_sequences(sequences_2, maxlen=max_len)
    x_3 = pad_sequences(sequences_3, maxlen=max_len)
    y = np.zeros((num_data,), dtype=np.float32)
    y[5331:] = np.ones((5331,), dtype=np.float32)
    indice = np.arange(num_data)
    np.random.shuffle(indice)
    x_1=x_1[indice]
    x_2 = x_2[indice]
    x_3 = x_3[indice]
    y=y[indice]


    return x_1,x_2,x_3, y


def get_model():
    input_1 = Input(shape=(max_len,))
    embedding1 = Embedding(num_words, word_dimension)(input_1)
    a = GlobalAveragePooling1D()(embedding1)
    a=Dense(word_dimension,activation='relu')(a)


    input_2 = Input(shape=(max_len,))
    embedding2 = Embedding(num_gram2, word_dimension)(input_2)
    b = GlobalAveragePooling1D()(embedding2)
    b = Dense(word_dimension, activation='relu')(b)


    input_3 = Input(shape=(max_len,))
    embedding3 = Embedding(num_gram3, word_dimension)(input_3)
    c = GlobalAveragePooling1D()(embedding3)
    c = Dense(word_dimension, activation='relu')(c)

    z=keras.layers.concatenate([a,b,c])

    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[input_1,input_2,input_3], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_1,x_2,x_3, y = get_input()
    # print(embedding_matrix[1,:])
    model = get_model()
    hist=model.fit([x_1,x_2,x_3], y, batch_size=32, epochs=10,
              validation_split=0.1)
    # print(hist.history)
    # print(max(hist.history['val_acc']))
    # acc=[]
    # for i in range(10):
    #     x=np.concatenate((x[-1066:],x[:-1066]),axis=0)
    #     y=np.concatenate((y[-1066:],y[:-1066]),axis=0)
    #     model = get_model(embedding_matrix)
    #     hist=model.fit(x, y, batch_size=32, epochs=10,
    #           validation_split=0.1)
    #     acc.append(max(hist.history['val_acc']))
    #     keras.backend.clear_session()
    # print(sum(acc)/10)

