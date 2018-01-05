# -*- coding: utf-8 -*-
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
import codecs
import math
import tensorflow as tf


MAX_DOCUMENT_LENGTH = 400
vocab_size = 30000
num_train = 75000
num_eval = 25000
Embed_dimension = 400
HIDDEN_SIZE = 500
batch_size = 32
num_sampled = 512

phrase = 'train'


def prepare_train():
    print("prepare training data")
    f = codecs.open('../../../temp/imdb/keras_code/utils/texts.pkl', 'rb')
    text1 = pickle.load(f)
    text1 = text1[:25000]
    f.close()
    f = codecs.open('../../../temp/imdb/keras_code/utils/texts_unsup.pkl', 'rb')
    text2 = pickle.load(f)
    f.close()
    texts = text1 + text2

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(texts)
    sequence = tokenizer.texts_to_sequences(texts)
    sequence_pad = pad_sequences(sequence, maxlen=MAX_DOCUMENT_LENGTH+1, dtype=np.int32, padding='post',
                                 truncating='post')
    seq_len = []
    for i in range(len(sequence)):
        r = len(sequence[i])
        if r<MAX_DOCUMENT_LENGTH:
            seq_len.append(r)
        else:
            seq_len.append(MAX_DOCUMENT_LENGTH)
    x_1 = sequence_pad[:, :-1]

    y_ = sequence_pad[:, 1:]
    return x_1, seq_len, y_




x, x_len, y = prepare_train()


start = 0
def get_input():
    global start

    if phrase is 'train':
        num_all = num_train
    else:
        num_all = num_eval
    if start + batch_size < num_all:
        x_place = x[start:start + batch_size]
        x_l = x_len[start:start + batch_size]

        yp = y[start:start + batch_size]
        start += batch_size
    else:
        x_place = np.concatenate((x[start:], x[:start - num_all + 32]), axis=0)
        x_l = np.concatenate((x_len[start:], x_len[:start - num_all + 32]))
        yp = np.concatenate((y[start:], y[:start - num_all + 32]))
        start = start - num_all + 32
    return x_place, x_l, yp


def train():
    x_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, MAX_DOCUMENT_LENGTH))
    x_len_place = tf.placeholder(dtype=tf.int64, shape=(batch_size,))
    y_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, MAX_DOCUMENT_LENGTH))
    with tf.device("/cpu:0"):
        embedding_word = tf.Variable(tf.random_uniform([vocab_size, Embed_dimension], -0.5, 0.5))
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, HIDDEN_SIZE],
                                stddev=1.0 / math.sqrt(HIDDEN_SIZE)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

        input = tf.nn.embedding_lookup(embedding_word, x_place)
    byte_list = tf.unstack(input, axis=1)
    with tf.variable_scope("myrnn"):
        cell = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, sequence_length=x_len_place, dtype=tf.float32)

    logits = tf.reshape(tf.stack(output, axis=1), [-1, HIDDEN_SIZE])
    y_lable = tf.reshape(y_place, [-1, 1])
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y_lable,
                       inputs=logits,
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
    for i in range(100):
        x_1, x_l, _y = get_input()
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _loss, _ = sess.run([loss, optimizer], feed_dict={x_place: x_1, x_len_place:x_l, y_place: _y})
        if i % 50 == 0:
            print(i, " loss ", _loss)

    saver.save(sess, '../../../temp/imdb/tf_code/lm_pretrain/lm_pretrain.ckpt')



train()
