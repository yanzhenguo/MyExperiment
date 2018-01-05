# -*- coding: utf-8 -*-
import pickle
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense,AveragePooling1D, TimeDistributed, \
    Conv2D,GlobalAveragePooling1D, GRU, MaxPooling1D

document_length =1024
word_length=32
char_embedding = 1600
root_dir = '../../../'

chars = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/|_#$%^&*~`+=<>()[]{} '
char_set = set(chars)
char_dic = {}
for i in range(len(chars)):
    char_dic[chars[i]] = i + 2


def word2num(word):
    '''将一个单词转化为字符id序列'''
    result = []
    word = word.lower()
    for i in range(word_length):
        if i < len(word):
            if word[i] in char_set:
                result.append(char_dic[word[i]])
            else:
                result.append(1)
        else:
            result.append(0)
    return result

def doc2num(doc):
    '''将一篇文档转化为字符id序列'''
    result=[]
    doc=doc.lower()
    for i in range(document_length):
        if i<len(doc):
            if doc[i] in char_set:
                result.append(char_dic[doc[i]])
            else:
                result.append(1)
        else:
            result.append(0)
    return result

def generate_input():
    '''将预料转化为输入矩阵，保存到文件中作为缓存'''
    pad_word = [0] * word_length
    with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
        texts = pickle.load(f)
    wordTokenizer = WordPunctTokenizer()
    new_text = []
    for i in range(len(texts)):
        doc = texts[i]
        # tokens = wordTokenizer.tokenize(doc)
        # new_tokens = []
        # temp = 0
        # for token in tokens:
        #     new_tokens+=word2num(token)
        #     temp += 1
        #     if temp >= word_length:
        #         break
        # if temp < document_length:
        #     for j in range(document_length - temp):
        #         new_tokens+=pad_word
        # new_text.append(new_tokens)
        new_text.append(doc2num(doc))
    x_train = np.asarray(new_text[:25000])
    x_test = np.asarray(new_text[25000:])
    print(x_train.shape)

    np.save(root_dir + "temp/imdb/keras_code/cnn-character/x_train.npy", x_train)
    np.save(root_dir + "temp/imdb/keras_code/cnn-character/x_test.npy", x_test)


def get_input():
    '''从文件中加载输入矩阵，打乱顺序后返回'''
    x_train = np.load(root_dir + "temp/imdb/keras_code/cnn-character/x_train.npy")
    x_test = np.load(root_dir + "temp/imdb/keras_code/cnn-character/x_test.npy")
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
    return x_train, y_train, x_test, y_test


def get_mdoel():
    '''构建keras模型'''
    input_1 = Input((document_length,))
    init_method = keras.initializers.Orthogonal()
    embedding1 = Embedding(len(chars)+2, char_embedding)(input_1)
    # x=TimeDistributed(Conv1D(filters=500, kernel_size=3, strides=1, activation='relu'))(embedding1)
    # x = TimeDistributed(Conv1D(filters=250, kernel_size=3, strides=1))(x)
    # x=TimeDistributed(GlobalAveragePooling1D())(x)
    x = Conv1D(filters=300, kernel_size=3, strides=1)(embedding1)
    # x=GRU(300)(x)
    x=MaxPooling1D(pool_size=3,strides=3)(x)
    x = Conv1D(filters=500, kernel_size=3, strides=1)(x)
    # x = MaxPooling1D(pool_size=2, strides=2)(x)
    # x = Conv1D(filters=800, kernel_size=3, strides=1)(x)
    # x = AveragePooling1D(pool_size=6, strides=1, padding='valid')(embedding1)
    x = GlobalMaxPooling1D()(x)
    # x=GlobalAveragePooling1D()(x)
    # x=GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_1, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # generate_input()
    x_train, y_train, x_test, y_test=get_input()
    model=get_mdoel()
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

