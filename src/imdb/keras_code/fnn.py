# -*- coding: utf-8 -*-

# 每个词赋予一个权重，用输入文本所有词的权重之和作逻辑回归，进行分类。

import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector, SpatialDropout1D,Activation,Reshape
from keras import regularizers,constraints


num_words = 80000
num_ngram=5000000
max_len = 500
root_dir = '../../../'

def get_input():
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters=''
    tokenizer.fit_on_texts(texts[:25000])
    print('there are %d words' % (len(tokenizer.word_index)))
    # word_index=tokenizer.word_index
    # index_word=dict([(index,word) for word, index in word_index.items()])
    sequences = tokenizer.texts_to_sequences(texts)
    new_text=[]
    for seq in sequences:
        t=set()
        for i in range(len(seq)):
            t.add('%d'%(seq[i]))
        for i in range(len(seq)-1):
            t.add('%d_%d'%(seq[i],seq[i+1]))
        for i in range(len(seq)-2):
            t.add('%d_%d_%d'%(seq[i],seq[i+1],seq[i+2]))
        new_text.append(' '.join(list(t)))

    tokenizer2=Tokenizer(num_words=num_ngram)
    tokenizer2.filters=''
    tokenizer2.fit_on_texts(new_text[:25000])
    sequences2=tokenizer2.texts_to_sequences(new_text)
    print('there are %d ngrams'%(len(tokenizer2.word_index)))

    x = pad_sequences(sequences2, maxlen=max_len)
    x_train = x[:25000]
    x_test = x[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    # print(x_train[0])
    return x_train, y_train, x_test, y_test

def get_model():
    word_embed=0.05*np.random.rand(num_ngram,1)-0.05
    word_embed[0]=0
    #print(word_embed[:50])

    main_input = Input(shape=(max_len,))
    embedding1 = Embedding(num_ngram, 1,
                           weights=[word_embed],
                           #embeddings_regularizer=regularizers.l2(0.01)
                           # embeddings_constraint=MaxNorm(5)
                           )(main_input)
    # x=SpatialDropout1D(0.3)(embedding1)
    # print(embedding1)
    x=Reshape((max_len,))(embedding1)
    # print(x)
    output=Dense(1,
            weights=[np.ones([max_len,1]),np.zeros([1])],
            trainable=False,
            activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_input()
    model = get_model()
    model.fit([x_train], y_train, batch_size=32, epochs=10,shuffle=True,
              validation_data=([x_test], y_test))
    # word_weight=model.get_layer('embedding_1').get_weights()[0]
    # word_weight_dict=dict()
    # for i in range(1, num_words):
    #     word_weight_dict[index_word[i]]=word_weight[i,0]
    #
    # word_weight_sort=sorted(word_weight_dict.items(),key=lambda item:item[1])
    #
    # with open('/home/yan/word_weights.txt','w') as f:
    #     for i in range(num_words-1):
    #         f.write('%-20s    %.3f\n'%(word_weight_sort[i][0],word_weight_sort[i][1]))