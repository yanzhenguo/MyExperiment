# -*- coding: utf-8 -*-

# 每个词赋予一个权重，用输入文本所有词的权重之和作逻辑回归，进行分类。

import pickle
import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, \
    BatchNormalization, RepeatVector, SpatialDropout1D, Activation, Reshape, Permute,Lambda
from keras import regularizers, constraints

num_words = 80000
num_ngram = 500000
max_len = 100
num_train = 120000
num_test = 76000
num_class=4
root_dir = '../../'


def get_input():
    with open(root_dir + "temp/ag/extract_data/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts[:num_train])
    print('there are %d words' % (len(tokenizer.word_index)))
    sequences = tokenizer.texts_to_sequences(texts)
    # print('average length is %d'%(np.sum([len(s) for s in sequences])/len(sequences)))
    new_text = []
    for seq in sequences:
        t = []
        for i in range(len(seq)):
            t.append('%d' % (seq[i]))
        for i in range(len(seq) - 1):
            t.append('%d_%d' % (seq[i], seq[i + 1]))
        # for i in range(len(seq) - 2):
        #     t.append('%d_%d_%d' % (seq[i], seq[i + 1], seq[i + 2]))
        new_text.append(' '.join(t))

    tokenizer2 = Tokenizer(num_words=num_ngram)
    tokenizer2.filters = ''
    tokenizer2.fit_on_texts(new_text[:num_train])
    sequences2 = tokenizer2.texts_to_sequences(new_text)
    print('there are %d ngrams' % (len(tokenizer2.word_index)))

    x = pad_sequences(sequences2, maxlen=max_len)
    x_train = x[:num_train]
    x_test = x[num_train:]
    y = np.load(root_dir + 'temp/ag/extract_data/label.npy')
    y_train = to_categorical(y[:num_train])
    y_test = to_categorical(y[num_train:])

    return x_train, y_train, x_test, y_test


def get_model():
    word_embed = 0.05 * np.random.rand(num_ngram, num_class) - 0.05

    main_input = Input(shape=(max_len,))
    embedding1 = Embedding(num_ngram, num_class,
                           weights=[word_embed],
                           )(main_input)
    x=Lambda(lambda x:K.sum(x,1))(embedding1)
    output = Activation('softmax')(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_input()
    model = get_model()
    model.fit([x_train], y_train, batch_size=256, epochs=10, shuffle=True,
              validation_data=([x_test], y_test))

# 准确率 92.70%