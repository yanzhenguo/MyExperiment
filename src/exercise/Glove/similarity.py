# coding=utf-8

# 找出与指定词在向量空间最接近的前n个词

import numpy as np
import codecs
import pickle

wordX = np.load('/home/yan/my_datasets/word2vec/word2vec.npy')
norm_wordX = 1 / (np.sum(wordX ** 2, axis=1) ** 0.5)
with codecs.open('/home/yan/my_datasets/word2vec/wordsInWord2vec.pkl', 'rb') as f:
    allwords = pickle.load(f)


def get_word_index(word):
    for i, w in enumerate(allwords):
        if w == word:
            return i


def find_near(word, num=10):
    word_index = get_word_index(word)
    word_norm = np.sum(wordX[word_index] ** 2) ** 0.5
    cos_sim1 = np.reshape(np.dot(wordX, np.reshape(wordX[word_index], [300, 1])), [3000000])
    cos_sim = cos_sim1 * norm_wordX
    max_index = np.argsort(cos_sim)
    for i in range(num):
        print(allwords[max_index[-i - 1]], ' ', cos_sim[max_index[-i - 1]] / word_norm)


if __name__ == '__main__':
    find_near('good')
