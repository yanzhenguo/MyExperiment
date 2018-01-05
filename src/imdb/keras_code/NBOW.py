# -*- coding: utf-8 -*-
import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector, SpatialDropout1D


num_words = 20000
max_len = 500
word_dimension = 500
root_dir = '../../../'


def get_input():
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts[0:25000])
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    embeddings_index = {}
    wordX = np.load(open("/home/yan/my_datasets/glove/embedding.300d.npy", mode='rb'))
    allwords = pickle.load(open("/home/yan/my_datasets/glove/words.pkl", mode='rb'))
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    # word_index = tokenizer.word_index
    # sequences = []
    # for i in range(50000):
    #     t = []
    #     tokens = texts[i].lower().split(' ')
    #     for j in range(len(tokens)):
    #         index = word_index.get(tokens[j], 0)
    #         if index < num_words:
    #             t.append(index)
    #         else:
    #             t.append(0)
    #     sequences.append(t)
    x = pad_sequences(sequences, maxlen=max_len)
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


def get_model(embedding_matrix):
    main_input = Input(shape=(max_len,))
    init_method = keras.initializers.Orthogonal()
    embedding1 = Embedding(num_words, 300, weights=[embedding_matrix])(main_input)
    # embedding1 = Embedding(num_words, 300)(main_input)
    # x=SpatialDropout1D(0.3)(embedding1)
    # x=Dropout(rate=0.3)(embedding1)
    x=GlobalAveragePooling1D()(embedding1)
    x = Dense(300, activation='relu')(x)
    x = Dense(300, activation='relu')(x)
    # embedding2 = Embedding(num_words, 300, weights=[embedding_matrix])(main_input)
    y=GlobalMaxPooling1D()(embedding1)

    z=keras.layers.concatenate([x,y])
    output = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=main_input, outputs=output)
    # ada=keras.optimizers.Adagrad(lr=0.005)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit([x_train], y_train, batch_size=32, epochs=10,
              validation_data=([x_test], y_test))
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, embedding_matrix = get_input()
    model = get_model(embedding_matrix)


    # for i in range(1):
    #     print("embedding dimension: ", embedding_dimension)
    #     model = get_model()
    #     model.fit([x_train], y_train, batch_size=32, epochs=10,
    #               validation_data=([x_test], y_test))
    #     keras.backend.clear_session()
    #     embedding_dimension += 100