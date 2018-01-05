# coding=utf-8

'''in this script, different n-grams share the same word embedding'''

import numpy as np
import codecs
import pickle
from nltk.tokenize import WordPunctTokenizer

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D

num_words = 30000
max_len = 500
embed_dimension=800

# prepare data
f = codecs.open('../../../temp/imdb/keras_code/utils/texts.pkl', 'rb')
text = pickle.load(f)
f.close()

# newText=[]
# for sentence in text:
#     t=[]
#     words = WordPunctTokenizer().tokenize(sentence)
#     for word in words:
#         if word.isalpha():
#             t.append(word)
#     newText.append(' '.join(t))

tokenizer = Tokenizer(num_words=num_words)
tokenizer.filters = ''
tokenizer.fit_on_texts(text[:25000])
sequence = tokenizer.texts_to_sequences(text)

# gram_dict={}
# for i in range(25000):
#     for j in range(len(sequence[i])-2):
#         if tuple(sequence[i][j:j+3]) in gram_dict:
#             gram_dict[tuple(sequence[i][j:j+3])]+=1
#         else:
#             gram_dict[tuple(sequence[i][j:j+3])]=1
# newSequence = []
# for i in range(len(sequence)):
#     t=[]
#     for j in range(len(sequence[i])-2):
#         #if tuple(sequence[i][j:j+3]) in gram_dict and gram_dict[tuple(sequence[i][j:j+3])]>0:
#         t+=sequence[i][j:j+3]
#     newSequence.append(t)

sequence_pad = pad_sequences(sequence, maxlen=max_len, dtype=np.int32, padding='post',truncating='post')

Xtrain = sequence_pad[:25000]
Xtest = sequence_pad[25000:]
y_train = np.zeros((25000,), dtype=np.int8)
y_test = np.zeros((25000,), dtype=np.int8)
y_train[12500:25000] = np.ones((12500,), dtype=np.int8)
y_test[12500:25000] = np.ones((12500,), dtype=np.int8)
indice = np.arange(25000)
np.random.shuffle(indice)
Xtrain=Xtrain[indice]
Xtest=Xtest[indice]
Ytrain=y_train[indice]
Ytest=y_test[indice]

input = Input(shape=(max_len,))

embedding = Embedding(num_words, embed_dimension, input_length=max_len, embeddings_initializer='normal')(input)
x = AveragePooling1D(pool_size=3, strides=1)(embedding)
# x=GlobalMaxPooling1D()(x)
#x=GlobalAveragePooling1D()(x)

y=AveragePooling1D(pool_size=2,strides=1)(embedding)
# y=GlobalMaxPooling1D()(y)
#y=GlobalAveragePooling1D()(y)

#p=GlobalAveragePooling1D()(embedding)
#p=GlobalMaxPooling1D()(embedding)

z=keras.layers.concatenate([x,y],axis=1)
z=GlobalMaxPooling1D()(z)
# z=keras.layers.maximum([x,y])
# x=Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, batch_size=32, epochs=10, validation_data=(Xtest, Ytest))
# model.save("../../../temp/imdb/keras_code/bowCNN_same/bowCNN_same.model.h5")
