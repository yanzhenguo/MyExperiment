# -*- coding: utf-8 -*-
import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector


num_words = 15000
max_len = 51
num_data=10662
num_train=9596
word_dimension = 600
root_dir = '../../'


def get_input():
    with open(root_dir + "temp/RT/util/text.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

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
    y = np.zeros((num_data,), dtype=np.float32)
    y[5331:] = np.ones((5331,), dtype=np.float32)
    indice = np.arange(num_data)
    np.random.shuffle(indice)
    x=x[indice]
    y=y[indice]


    return x, y


def get_model():
    main_input = Input(shape=(max_len,))
    init_method = keras.initializers.Orthogonal()
    embedding1 = Embedding(num_words, word_dimension)(main_input)
    # x=Dropout(rate=0.3)(embedding1)
    x = AveragePooling1D(pool_size=3, strides=1, padding='valid')(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x, y = get_input()
    acc=[]
    for i in range(10):
        x=np.concatenate((x[-1066:],x[:-1066]),axis=0)
        y=np.concatenate((y[-1066:],y[:-1066]),axis=0)
        model = get_model()
        hist=model.fit(x, y, batch_size=32, epochs=10,
              validation_split=0.1)
        acc.append(max(hist.history['val_acc']))
        keras.backend.clear_session()
    print(sum(acc) / 10)
    # for i in range(1):
    #     print("embedding dimension: ", embedding_dimension)
    #     model = get_model()
    #     model.fit([x_train], y_train, batch_size=32, epochs=10,
    #               validation_data=([x_test], y_test))
    #     keras.backend.clear_session()
    #     embedding_dimension += 100
