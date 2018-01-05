# -*- coding: utf-8 -*-
import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D, Concatenate

num_words = 20000
max_len = 500
f = codecs.open('../temp/texts.pkl', 'rb')
# f = codecs.open('texts.pkl','rb')
texts = pickle.load(f)
f.close()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.filters=''
tokenizer.fit_on_texts(texts[0:25000])
# sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
sequences = []
for i in range(50000):
    t = []
    tokens = texts[i].lower().split(' ')
    for j in range(len(tokens)):
        index = word_index.get(tokens[j], 0)
        if index < num_words:
            t.append(index)
        else:
            t.append(0)
    sequences.append(t)

print('Found %s unique tokens.' % len(word_index))

data1 = pad_sequences(sequences[0:25000], maxlen=max_len)
data2 = pad_sequences(sequences[25000:50000], maxlen=max_len)
Ytrain = np.zeros((25000,), dtype=np.float32)
Ytest = np.zeros((25000,), dtype=np.float32)
Ytrain[12500:25000] = np.ones((12500,), dtype=np.float32)
Ytest[12500:25000] = np.ones((12500,), dtype=np.float32)

Xtrain1 = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
Xtest1 = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
for i in range(25000):
    for j in range(max_len - 2):
        Xtrain1[i, j * 3] = data1[i, j]
        Xtrain1[i, j * 3 + 1] = data1[i][j + 1] + num_words
        Xtrain1[i, j * 3 + 2] = data1[i][j + 2] + num_words * 2
for i in range(25000):
    for j in range(max_len - 2):
        Xtest1[i, j * 3] = data2[i, j]
        Xtest1[i, j * 3 + 1] = data2[i][j + 1] + num_words
        Xtest1[i, j * 3 + 2] = data2[i][j + 2] + num_words * 2

Xtrain2 = np.zeros((25000, (max_len - 1) * 2), dtype=np.int)
Xtest2 = np.zeros((25000, (max_len - 1) * 2), dtype=np.int)
for i in range(25000):
    for j in range(max_len - 1):
        Xtrain2[i, j * 2] = data1[i, j]
        Xtrain2[i, j * 2 + 1] = data1[i][j + 1] + num_words
for i in range(25000):
    for j in range(max_len - 1):
        Xtest2[i, j * 2] = data2[i, j]
        Xtest2[i, j * 2 + 1] = data2[i][j + 1] + num_words

indice1 = np.arange(25000)
np.random.shuffle(indice1)
Xtrain1 = Xtrain1[indice1]
Xtrain2 = Xtrain2[indice1]
Ytrain = Ytrain[indice1]

indice2 = np.arange(25000)
np.random.shuffle(indice2)
Xtest1 = Xtest1[indice2]
Xtest2 = Xtest2[indice2]
Ytest = Ytest[indice2]
print('begin to build model ...')
input1 = Input(shape=((max_len - 2) * 3,))
embedding1 = Embedding(num_words * 3, 800, embeddings_initializer=keras.initializers.Orthogonal())(input1)
x = AveragePooling1D(pool_size=3)(embedding1)
x = GlobalMaxPooling1D()(x)

input2 = Input(shape=((max_len - 1) * 2,))
embedding2 = Embedding(num_words * 2, 500, embeddings_initializer=keras.initializers.Orthogonal())(input2)
y = AveragePooling1D(pool_size=2, strides=2)(embedding2)
y = GlobalMaxPooling1D()(y)

z = Concatenate()([x, y])
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit([Xtrain1, Xtrain2], Ytrain, batch_size=32, epochs=50, validation_data=([Xtest1, Xtest2], Ytest))

# model=Model(inputs=input2,outputs=output)
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit([Xtrain2], Ytrain,batch_size=32,epochs=50,validation_data=([Xtest2], Ytest))
