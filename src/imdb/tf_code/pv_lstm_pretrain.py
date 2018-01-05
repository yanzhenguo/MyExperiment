# -*- coding: utf-8 -*-
'''
    使用RNN得到文档向量，利用此向量预测这个文档中的词，思想来源于paragraph vector(pv).
'''

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
import codecs
import math
import tensorflow as tf

MAX_DOCUMENT_LENGTH = 400
vocab_size = 30000
num_train = 25000
Embed_dimension = 300
HIDDEN_SIZE = 500
batch_size = 256
num_sampled = 256


def prepare_train():
    print("prepare training data")
    f = codecs.open('../../../temp/imdb/keras_code/utils/texts.pkl', 'rb')
    text1 = pickle.load(f)
    text1 = text1[:25000]
    f.close()
    # f = codecs.open('../../../temp/imdb/keras_code/utils/texts_unsup.pkl', 'rb')
    # text2 = pickle.load(f)
    # f.close()
    # texts = text1 + text2
    texts = text1

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequence = tokenizer.texts_to_sequences(texts)
    sequence_pad = pad_sequences(sequence, maxlen=MAX_DOCUMENT_LENGTH, dtype=np.int32, padding='post')
    seq_len = []
    for i in range(len(sequence)):
        r = len(sequence[i])
        if r < MAX_DOCUMENT_LENGTH:
            seq_len.append(r)
        else:
            seq_len.append(MAX_DOCUMENT_LENGTH)
    return sequence_pad, seq_len


x, x_len = prepare_train()
doc_index = 0
word_index = 0


def get_input():
    global doc_index
    global word_index
    x1 = np.zeros((batch_size, MAX_DOCUMENT_LENGTH))
    y = np.zeros((batch_size,))
    x1_len = np.zeros((batch_size,))
    for i in range(batch_size):
        x1[i] = x[doc_index]
        y[i] = x[doc_index, word_index]
        x1_len[i] = x_len[doc_index]
        word_index += 1
        if word_index >= x_len[doc_index]:
            word_index = 0
            doc_index += 1
            if doc_index >= num_train:
                doc_index = 0
    y = np.reshape(y, (batch_size, 1))
    return x1, x1_len, y


def train():
    x_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, MAX_DOCUMENT_LENGTH))
    x_len_place = tf.placeholder(dtype=tf.int64, shape=(batch_size,))
    y_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    with tf.device("/cpu:0"):
        embedding_word = tf.Variable(tf.random_uniform([vocab_size, Embed_dimension], -0.5, 0.5))
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, HIDDEN_SIZE],
                                stddev=1.0 / math.sqrt(HIDDEN_SIZE)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

        input_1 = tf.nn.embedding_lookup(embedding_word, x_place)
    byte_list = tf.unstack(input_1, axis=1)
    with tf.variable_scope("myrnn"):
        cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, sequence_length=x_len_place, dtype=tf.float32)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y_place,
                       inputs=encoding,
                       num_sampled=num_sampled,
                       num_classes=vocab_size))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()

    val_to_save = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="myrnn")
    val_to_save = val_to_save.append(embedding_word)
    saver = tf.train.Saver(val_to_save)

    sess = tf.Session()
    sess.run(init)
    # save_resotre = tf.train.Saver()
    # save_resotre.restore(sess,'../../../temp/imdb/tf_code/lm_pretrain/lm_pretrain.ckpt')
    for i in range(50000):
        x_1, x_l, _y = get_input()
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _loss, _ = sess.run([loss, optimizer], feed_dict={x_place: x_1, x_len_place: x_l, y_place: _y})
        if i % 100 == 0:
            print(i, " loss ", _loss)

    saver.save(sess, '../../../temp/imdb/tf_code/lm_pretrain/lm_pretrain.ckpt')


if __name__ == '__main__':
    train()
