# -*- coding: utf-8 -*-
import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector,GRU

num_words=30000
max_len=500
word_dimension=300
root_dir = '../../../'


def get_input():
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts[0:25000])
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=max_len, truncating='post')

    word_index = tokenizer.word_index
    embeddings_index = {}
    wordX = np.load('/home/yan/my_datasets/word2vec/word2vec.npy')
    f = open('/home/yan/my_datasets/word2vec/wordsInWord2vec.pkl', 'rb')
    allwords = pickle.load(f)
    f.close()
    for i in range(3000000):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    x_train = x[:25000]
    x_test = x[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    indice = np.arange(25000)
    np.random.shuffle(indice)
    x_train = x_train[indice]
    x_test = x_test[indice]
    y_train = y_train[indice]
    y_test = y_test[indice]

    return x_train, y_train, x_test, y_test, embedding_matrix

def get_model(embed_word2vec):
    input_1=Input(shape=(max_len,))
    embedding_1=Embedding(input_dim=num_words,
                          output_dim=word_dimension,
                          weights=[embed_word2vec])(input_1)
    x=GRU(word_dimension)(embedding_1)
    # x=Bidirectional(GRU(word_dimension),merge_mode='concat')(embedding_1)
    output_1=Dense(1,activation='sigmoid')(x)
    model=Model(inputs=[input_1],outputs=[output_1])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


if __name__=='__main__':
    x_train, y_train, x_test, y_test, embedding_matrix=get_input()
    model=get_model(embedding_matrix)
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=([x_test], [y_test]))