# -*- coding: utf-8 -*-
'''
    将n-gram embedding作为lstm的输入
'''

import pickle
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, AveragePooling1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout,\
    BatchNormalization, RepeatVector,GRU

num_words=30000
max_len=400
word_dimension=300
num_ngram=400000
root_dir = '../../../'

def get_input():
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    word_index=pickle.load(open(root_dir+'temp/imdb/tf_code/pv_ngram/word_index.pkl','rb'))
    ngram_index = pickle.load(open(root_dir + 'temp/imdb/tf_code/pv_ngram/ngram_index.pkl','rb'))

    sequence=[]
    for sentence in texts:
        t_s=[]
        for token in sentence.split(' '):
            if token in word_index:
                t_s.append(str(word_index[token]))
        sequence.append(t_s)

    new_sequence=[]
    for seq in sequence:
        t_s=[]
        for i in range(len(seq)-2):
            s='_'.join(seq[i:i+3])
            if s in ngram_index and ngram_index[s]<=num_ngram:
                t_s.append(ngram_index[s])
        new_sequence.append(t_s)
    new_sequence=pad_sequences(new_sequence,maxlen=max_len)

    x_train=new_sequence[:25000]
    x_test=new_sequence[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    return x_train,y_train,x_test,y_test

def get_model():
    ngram_embed=np.load(root_dir + "temp/imdb/tf_code/pv_ngram/ngram_embedding.npy")
    ngram_embedding=np.random.randn(num_ngram+1,word_dimension)
    ngram_embedding[1:]=ngram_embed

    input_1=Input(shape=(max_len,))
    embedding_1=Embedding(input_dim=num_ngram+1,
                          output_dim=word_dimension,
                          weights=[ngram_embedding],
                          trainable=True)(input_1)
    x=GRU(word_dimension)(embedding_1)
    # x=Bidirectional(GRU(word_dimension),merge_mode='concat')(embedding_1)
    output_1=Dense(1,activation='sigmoid')(x)
    model=Model(inputs=[input_1],outputs=[output_1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__=='__main__':
    x_train, y_train, x_test, y_test=get_input()
    model=get_model()
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              shuffle=True,
              validation_data=([x_test], [y_test]))
