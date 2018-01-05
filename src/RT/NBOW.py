# -*- coding: utf-8 -*-
import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector, SpatialDropout1D


num_words = 30000
max_len = 51
num_data=10662
num_train=9596
word_dimension = 300
root_dir = '../../'


def get_input():
    with open(root_dir + "temp/RT/util/text.pkl", 'rb') as f:
        texts = pickle.load(f)
    print('average length of sample is: ',sum([len(seq.split(' ')) for seq in texts])/len(texts))
    print('max length of sample is: ',max([len(seq.split(' ')) for seq in texts]))
    print('min length of sample is: ', min([len(seq.split(' ')) for seq in texts]))

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('there are %d words' % (len(word_index)))
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
    y = np.zeros((num_data,), dtype=np.float32)
    y[5331:] = np.ones((5331,), dtype=np.float32)
    indice = np.arange(num_data)
    np.random.shuffle(indice)
    x=x[indice]
    y=y[indice]


    return x, y, embedding_matrix


def get_model(embedding_matrix):
    main_input = Input(shape=(max_len,))
    init_method = keras.initializers.Orthogonal()
    # embedding1 = Embedding(num_words, word_dimension,weights=[embedding_matrix])(main_input)
    # a=SpatialDropout1D(0.3)(embedding1)
    # x=Dropout(rate=0.3)(embedding1)
    # x = GlobalAveragePooling1D()(a)
    # x=Dense(300,activation='relu')(x)
    # x=Dense(300,activation='relu')(x)
    # x = Dense(300, activation='relu')(x)

    init_2=keras.initializers.RandomUniform(minval=-0.05,maxval=0.05)
    embedding2 = Embedding(num_words, 500,embeddings_initializer=init_2)(main_input)
    # b=SpatialDropout1D(0.3)(embedding2)
    b=GlobalMaxPooling1D()(embedding2)


    # z=keras.layers.concatenate([x,b])
    # z = Dense(300, activation='relu')(z)
    # z = Dense(300, activation='relu')(z)

    output = Dense(1, activation='sigmoid')(b)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x, y, embedding_matrix = get_input()
    # print(embedding_matrix[1,:])
    # model = get_model()
    # hist=model.fit(x, y, batch_size=32, epochs=10,
    #           validation_split=0.1)
    # print(hist.history)
    # print(max(hist.history['val_acc']))
    acc=[]
    for i in range(10):
        x=np.concatenate((x[-1066:],x[:-1066]),axis=0)
        y=np.concatenate((y[-1066:],y[:-1066]),axis=0)
        model = get_model(embedding_matrix)
        hist=model.fit(x, y, batch_size=32, epochs=10,
              validation_split=0.1)
        acc.append(max(hist.history['val_acc']))
        keras.backend.clear_session()
    print(sum(acc)/10)

