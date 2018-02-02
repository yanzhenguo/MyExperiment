# -*- coding: utf-8 -*-

# 利用短语级别的样本进行训练

import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical

num_words=15000
maxlen = 50
batch_size = 32
embedding_dims = 300
nb_filter = 500
nb_epoch = 10
train_size = 6920
test_size = 1821
val_size = 872

temp_dir1='../../../temp/StanfortSentiment/binary/preprocess/'
temp_dir2='../../../temp/StanfortSentiment/binary/processTrain/'


def get_glove(word_index):
    embeddings_index = {}
    wordX = np.load('/home/yan/my_datasets/glove/embedding.300d.npy')
    with open('/home/yan/my_datasets/glove/words.pkl', 'rb') as f:
        allwords = pickle.load(f)
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_word2vec(word_index):
    embeddings_index = {}
    wordX = np.load('/home/yan/my_datasets/word2vec/word2vec.npy')
    with open('/home/yan/my_datasets/word2vec/words.pkl', 'rb') as f:
        allwords = pickle.load(f)
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_input():
    with open(temp_dir2+"train_sentence.pkl",'rb') as f:
        texts_train=pickle.load(f)
    with open(temp_dir1+"texts.pkl",'rb') as f:
        texts_testVal=pickle.load(f)[train_size:]
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts_train)
    sequences_train = tokenizer.texts_to_sequences(texts_train)
    sequences_test_val=tokenizer.texts_to_sequences(texts_testVal)
    word_index=tokenizer.word_index
    embedding_matrix=get_word2vec(word_index)
    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(sequences_train, maxlen=maxlen)
    x_test = pad_sequences(sequences_test_val[:test_size], maxlen=maxlen)
    x_val = pad_sequences(sequences_test_val[test_size:], maxlen=maxlen)

    y_train=np.load(temp_dir2+'Ytrain.npy')
    y_test=np.load(temp_dir1+"Ytest.npy")
    y_val=np.load(temp_dir1+'Yval.npy')

    return x_train,y_train,x_test,y_test,embedding_matrix

def bulid_model():
    x_train,y_train,x_test,y_test,embedding_matrix=get_input()
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dims,
                        input_length=maxlen,
                        weights=[embedding_matrix],
                        trainable=True))
    model.add(Convolution1D(filters=nb_filter,
                            kernel_size =3,
                            activation='relu'))
    model.add(GlobalMaxPooling1D())
    # model.add(Dense(100,activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              validation_data=(x_test, y_test))


if __name__=='__main__':
    bulid_model()