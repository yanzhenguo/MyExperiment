# -*- coding: utf-8 -*-
'''
    实现paragraph vector(pv).
'''

from keras.preprocessing.text import Tokenizer
from tensorflow.python.lib.io.file_io import FileIO

import numpy as np
import argparse
import os
import pickle
import math
import tensorflow as tf

vocab_size = 30000
Embed_dimension = 400
batch_size = 256
num_sampled = 256
tokenizer = None
FLAGS = None

def prepare_train():
    global tokenizer
    print("prepare training data")
    with FileIO(os.path.join(FLAGS.buckets, "texts.pkl"), mode='r+') as f:
        texts = pickle.load(f)[:25000]
    with FileIO(os.path.join(FLAGS.buckets, "texts_unsup.pkl"), mode='r+') as f:
        texts += pickle.load(f)

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequence = tokenizer.texts_to_sequences(texts)
    sum_words = sum([len(seq) for seq in sequence])
    print('there are %d words' % (sum_words))
    x = np.zeros((sum_words, 1), dtype=np.int32)
    y = np.zeros((sum_words, 1), dtype=np.int32)
    index = 0
    for i, seq in enumerate(sequence):
        for s in seq:
            x[index] = i
            y[index] = s
            index += 1
    indice = np.arange(sum_words)
    np.random.shuffle(indice)
    x = x[indice]
    y = y[indice]
    return x, y, sum_words


def get_input(x, y, sum_words, start):
    if start + batch_size < sum_words:
        start += batch_size
        return x[start-batch_size:start], y[start-batch_size:start], start

    else:
        # print('loop 2')
        # print(start, batch_size, sum_words)
        start = batch_size - sum_words + start
        return np.concatenate((x[start-batch_size:], x[:start]),axis=0), \
               np.concatenate((y[start-batch_size:], y[:start]),axis=0), start


def train():
    x_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    y_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    with tf.device("/cpu:0"):
        embedding_doc = tf.Variable(tf.random_uniform([75000, Embed_dimension], -0.5, 0.5))
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, Embed_dimension],
                                stddev=1.0 / math.sqrt(Embed_dimension)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

        input_1 = tf.nn.embedding_lookup(embedding_doc, x_place)
    input_2 = tf.reshape(input_1, [-1, Embed_dimension])
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y_place,
                       inputs=input_2,
                       num_sampled=num_sampled,
                       num_classes=vocab_size))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    x, y, sum_words = prepare_train()
    start = 0
    for i in range(4000000):
        x_1, _y, start = get_input(x, y, sum_words, start)
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _loss, _ = sess.run([loss, optimizer], feed_dict={x_place: x_1, y_place: _y})
        if i % 3000 == 0:
            print(i, " loss ", _loss)
    return embedding_doc.eval(sess)[:25000], nce_weights.eval(sess), nce_biases.eval(sess)


def prepare_evalute():
    global tokenizer
    print("prepare evaluation data")
    with FileIO(os.path.join(FLAGS.buckets, "texts.pkl"), mode='r+') as f:
        texts = pickle.load(f)[25000:]

    sequence = tokenizer.texts_to_sequences(texts)
    sum_words = sum([len(seq) for seq in sequence])
    print('there are %d words' % sum_words)
    x = np.zeros((sum_words, 1), dtype=np.int32)
    y = np.zeros((sum_words, 1), dtype=np.int32)
    index = 0
    for i, seq in enumerate(sequence):
        for s in seq:
            x[index] = i
            y[index] = s
            index += 1
    indice = np.arange(sum_words)
    np.random.shuffle(indice)
    x = x[indice]
    y = y[indice]
    return x, y, sum_words


def get_input_eval(x, y, sum_words, start):
    if start + batch_size < sum_words:
        start += batch_size
        return x[start-batch_size:start], y[start-batch_size:start], start

    else:
        start = batch_size - sum_words + start
        return np.concatenate((x[start-batch_size:], x[:start]),axis=0), \
               np.concatenate((y[start-batch_size:], y[:start]),axis=0), start


def evaluate(nce_weighting, nce_bias):
    x_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    y_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    with tf.device("/cpu:0"):
        embedding_doc = tf.Variable(tf.random_uniform([25000, Embed_dimension], -0.5, 0.5))
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, Embed_dimension],
                                stddev=1.0 / math.sqrt(Embed_dimension)), trainable=False)
        nce_biases = tf.Variable(tf.zeros([vocab_size]), trainable=False)

        input_1 = tf.nn.embedding_lookup(embedding_doc, x_place)
    input_2 = tf.reshape(input_1, [-1, Embed_dimension])
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y_place,
                       inputs=input_2,
                       num_sampled=num_sampled,
                       num_classes=vocab_size))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    ass_nce_wei = tf.assign(nce_weights, nce_weighting)
    ass_nce_bias = tf.assign(nce_biases, nce_bias)

    sess = tf.Session()
    sess.run([init])
    sess.run([ass_nce_wei, ass_nce_bias])
    x, y, sum_words = prepare_evalute()
    start = 0
    for i in range(3000000):
        x_1, _y, start = get_input_eval(x, y, sum_words, start)
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _loss, _ = sess.run([loss, optimizer], feed_dict={x_place: x_1, y_place: _y})
        if i % 3000 == 0:
            print(i, " loss ", _loss)
    return embedding_doc.eval(sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    traing_embed, nce_weight, nce_bias = train()
    test_embed = evaluate(nce_weight, nce_bias)
    np.save(FileIO(os.path.join(FLAGS.buckets, "docembed.npy"),'w+'), np.concatenate((traing_embed, test_embed), axis=0))
