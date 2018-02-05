# -*- coding: utf-8 -*-
import pickle
import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, \
    BatchNormalization, RepeatVector, SpatialDropout1D, Lambda, Activation, Reshape

num_words = 30000
max_len = 1000
word_dimension = 1000
root_dir = '../../../'


def get_input():
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(texts[:25000])
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
    x_train = x[:25000]
    x_test = x[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    return x_train, y_train, x_test, y_test


def get_model():
    weight = np.ones((word_dimension, 1), dtype=np.float)
    weight[int(word_dimension / 2):] = -1 * np.ones([int(word_dimension / 2), 1], dtype=np.float)

    main_input = Input(shape=(max_len,))
    init_method = keras.initializers.Orthogonal()
    # embedding1 = Embedding(num_words, word_dimension,embeddings_initializer=init_method)(main_input)
    embedding1 = Embedding(num_words, word_dimension)(main_input)
    # x=SpatialDropout1D(0.3)(embedding1)
    # x=Dropout(rate=0.3)(embedding1)
    x = AveragePooling1D(pool_size=4, strides=1, padding='valid')(embedding1)
    x = GlobalMaxPooling1D()(x)
    output = Dense(1,
                   weights=[weight, np.zeros([1])],
                   trainable=False,
                   activation='sigmoid')(x)

    # print(output)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


def analyze(model):
    # 从训练集中找到正向负向评分最高的n-gram
    print('analyze model')
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts[0:25000])
    word_index = tokenizer.word_index
    index_word = {word_index[key]: key for key in word_index.keys()}
    sequences = tokenizer.texts_to_sequences(texts)
    gram_3 = {}
    for seq in sequences:
        for i in range(len(seq) - 2):
            if tuple(seq[i:i + 3]) not in gram_3:
                gram_3[tuple(seq[i:i + 3])] = 0.5
    print('there are %d 3-grams' % (len(gram_3)))
    all_gram = list(gram_3.keys())
    all_gram_score = np.zeros((len(all_gram)))
    input_1 = np.zeros((1000, max_len))
    for i in range(6206):
        for j in range(1000):
            input_1[j][0:3] = all_gram[1000 * i + j][:]
        all_gram_score[i * 1000:(i + 1) * 1000] = np.reshape(model.predict(input_1), (1000,))
        if i % 100 == 0:
            print('current i is %d' % (i))

    for i in range(len(all_gram)):
        gram_3[all_gram[i]] = all_gram_score[i]
    gram_3_sort = sorted(gram_3.items(), key=lambda asd: asd[1], reverse=True)
    # print(gram_3_sort[:100])
    for i in range(20):
        n_gram = gram_3_sort[i][0]
        print(index_word[n_gram[0]] + ' ' + index_word[n_gram[1]] + ' ' + index_word[n_gram[2]])
    for i in range(20):
        n_gram = gram_3_sort[len(all_gram) - i - 403][0]
        print(index_word[n_gram[0]] + ' ' + index_word[n_gram[1]] + ' ' + index_word[n_gram[2]])


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_input()
    model = get_model()
    model.fit([x_train], y_train, batch_size=64, epochs=10, shuffle=True,
              validation_data=([x_test], y_test))
    # predict_result=model.predict(x_test,batch_size=25)
    # file_str=[]
    # # print(predict_result)
    # for i in range(25000):
    #     if predict_result[i]<0.5:
    #         file_str.append('0')
    #     else:
    #         file_str.append('1')
    # with open('/home/yan/bow-NNGN_predction.txt','w') as f:
    #     f.write('\n'.join(file_str))
    # for i in range(1):
    #     print("embedding dimension: ", embedding_dimension)
    #     model = get_model()
    #     model.fit([x_train], y_train, batch_size=32, epochs=10,
    #               validation_data=([x_test], y_test))
    #     keras.backend.clear_session()
    #     embedding_dimension += 100
    # analyze(model)
