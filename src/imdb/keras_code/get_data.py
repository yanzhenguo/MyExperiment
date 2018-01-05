# -*- coding: utf-8 -*-
import codecs
import pickle
import keras
import numpy as np
from keras.layers import Dense, GlobalMaxPooling1D, Input, Embedding, \
    AveragePooling1D, GlobalAveragePooling1D, Activation
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import util


def data_bow_cnn(num_words, max_len):
    # 将文本序列转化为id序列
    imdb = util.ImdbCorpus(num_words=num_words,max_len=max_len)
    Xtrain, Ytrain, Xtest, Ytest = imdb.get_input()
    util.shuffleData([Xtrain, Ytrain, Xtest, Ytest])
    return [Xtrain, Ytrain, Xtest, Ytest]


def data_bow(num_words=30000, max_len=500):
    # 将输入表示为bow,用fnn进行分类
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    [xtrain, ytrain, xtest, ytest] = imdb.get_input_bow()
    # util.shuffleData([xtrain, ytrain, xtest, ytest])
    return [xtrain, ytrain, xtest, ytest]


def data_bow_cnn2(num_words, max_len):
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len)
    Xtrain, Ytrain, Xtest, Ytest = imdb.get_input()

    model2 = load_model('../temp/bowCNN.model.h5')
    embedding_matrix2 = model2.get_layer('embedding_1').get_weights()[0]
    weights_dense = model2.get_layer('dense_1').get_weights()
    # embedding_matrix2 = np.load('../temp/embedding2.npy')
    embedding_matrix = imdb.get_word2vec_matrix()

    util.shuffleData([Xtrain, Ytrain, Xtest, Ytest])
    return [Xtrain, Ytrain, Xtest, Ytest, embedding_matrix2, weights_dense, embedding_matrix]


def data_cnn(num_words, max_len, use_word2vec=True, truncing='post',padding='pre'):
    imdb = util.ImdbCorpus(num_words=num_words, max_len=max_len,truncing=truncing,padding=padding)
    Xtrain, Ytrain, Xtest, Ytest = imdb.get_input()

    [Xtrain, Ytrain, Xtest, Ytest]=util.shuffleData([Xtrain, Ytrain, Xtest, Ytest])
    if use_word2vec:
        embedding_matrix = imdb.get_word2vec_matrix()
        return [Xtrain, Ytrain, Xtest, Ytest, embedding_matrix]
    else:
        return [Xtrain, Ytrain, Xtest, Ytest]