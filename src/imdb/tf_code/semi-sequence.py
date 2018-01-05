# -*- coding: utf-8 -*-
'''将文档向量作为循环神经网络的初始状态，训练网络根据词向量还原出该文档，从而得到文档向量'''
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
HIDDEN_SIZE = 500
batch_size = 32
num_sampled = 512

phrase = 'train'


def prepare_train():
    print("prepare training data")
    f = codecs.open('../temp/texts.pkl', 'rb')
    text1 = pickle.load(f)
    text1 = text1[:25000]
    f.close()
    f = codecs.open('../temp/texts_unsup.pkl', 'rb')
    text2 = pickle.load(f)
    f.close()
    texts = text1 + text2

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(texts)
    sequence = tokenizer.texts_to_sequences(texts)
    sequence_pad = pad_sequences(sequence, maxlen=MAX_DOCUMENT_LENGTH+1, dtype=np.int32, padding='post',
                                 truncating='post')

    x_1 = sequence_pad[:, :-1]
    x_2 = np.arange(num_train)
    x_2 = np.reshape(x_2, (num_train, 1))
    y_ = sequence_pad[:, 1:]
    return x_1, x_2, y_


def prepare_eval():
    print("prepare evaluating data")
    f = codecs.open('../temp/texts.pkl', 'rb')
    text1 = pickle.load(f)
    texts = text1[25000:]
    f.close()

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(texts)
    sequence = tokenizer.texts_to_sequences(texts)
    sequence_pad = pad_sequences(sequence, maxlen=MAX_DOCUMENT_LENGTH+1, dtype=np.int32, padding='post',
                                 truncating='post')

    x_1 = sequence_pad[:, :-1]
    x_2 = np.arange(num_eval)
    x_2 = np.reshape(x_2, (num_eval, 1))
    y_ = sequence_pad[:,1:]
    return x_1, x_2, y_


if phrase is 'train':
    x1, x2, y = prepare_train()
else:
    x1, x2, y = prepare_eval()

start = 0

def get_input():
    global start

    if phrase is 'train':
        num_all = num_train
    else:
        num_all = num_eval
    if start + batch_size < num_all:
        x1_place = x2[start:start + batch_size]
        x2_place = x1[start:start + batch_size]
        yp = y[start:start + batch_size]
        start += batch_size
    else:
        x1_place = np.concatenate((x2[start:], x2[:start - num_all + 32]), axis=0)
        x2_place = np.concatenate((x1[start:], x1[:start - num_all + 32]), axis=0)
        yp = np.concatenate((y[start:], y[:start - num_all + 32]))
        start = start - num_all + 32
    return x1_place, x2_place, yp


def train():
    x1_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    # x2_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, MAX_DOCUMENT_LENGTH))
    y_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, MAX_DOCUMENT_LENGTH))
    with tf.device("/cpu:0"):
        embedding_doc = tf.Variable(tf.random_uniform([num_train, HIDDEN_SIZE], -0.5, 0.5))
        # embedding_word = tf.Variable(tf.random_uniform([vocab_size, HIDDEN_SIZE], -0.5, 0.5))
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, HIDDEN_SIZE],
                                stddev=1.0 / math.sqrt(HIDDEN_SIZE)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

        input1 = tf.nn.embedding_lookup(embedding_doc, x1_place)
        # input2 = tf.nn.embedding_lookup(embedding_word, x2_place)
    input1_1 = tf.reshape(input1, [-1, HIDDEN_SIZE])
    # input = tf.concat([input1, input2], axis=1)
    # byte_list = tf.unstack(input, axis=1)
    cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
    print(cell)
    # input1_1 = tf.expand_dims(input1,1)
    outputs = []
    state = cell.zero_state(32, tf.float32)
    with tf.variable_scope("RNN"):
        for time_step in range(MAX_DOCUMENT_LENGTH):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            #(cell_output, state) = cell(tf.concat([input1_1,input2[:, time_step, :]],axis=1), state)
            (cell_output, state) = cell(input1_1, state)
            outputs.append(cell_output)
    logits = tf.reshape(tf.stack(axis=1, values=outputs), [-1, HIDDEN_SIZE])



    #output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, dtype=tf.float32)

    # logits = tf.reshape(tf.stack(output, axis=1), [-1, HIDDEN_SIZE])
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


    sess = tf.Session()
    sess.run(init)
    for i in range(10000):
        x_1, x_2, _y = get_input()
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, y_place: _y})
        if i % 50 == 0:
            print(i, " loss ", _loss)
    embed_doc = embedding_doc.eval(sess)
    np.save("../temp/embedding_train_doc.npy", embed_doc)

    saver = tf.train.Saver()
    saver.save(sess, '../temp/semi_sequence.ckpt')


def model_eval():
    x1_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    x2_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, MAX_DOCUMENT_LENGTH))
    y_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, MAX_DOCUMENT_LENGTH + 1))
    with tf.device("/cpu:0"):
        embedding_doc = tf.Variable(tf.random_uniform([num_train, HIDDEN_SIZE], -0.5, 0.5),trainable=True)
        embedding_word = tf.Variable(tf.random_uniform([vocab_size, HIDDEN_SIZE], -0.5, 0.5),trainable=False)
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, HIDDEN_SIZE],
                                stddev=1.0 / math.sqrt(HIDDEN_SIZE)), trainable=False)
        nce_biases = tf.Variable(tf.zeros([vocab_size]), trainable=False)

        input1 = tf.nn.embedding_lookup(embedding_doc, x1_place)
        input2 = tf.nn.embedding_lookup(embedding_word, x2_place)
    input = tf.concat([input1, input2], axis=1)
    byte_list = tf.unstack(input, axis=1)
    cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
    output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, dtype=tf.float32)
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
    # train_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # local_val = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    # for i in range(len(train_val)):
    #     if i != 0:
    #         local_val.append(train_val[i])
    # init = tf.global_variables_initializer()
    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, '../temp/semi_sequence.ckpt')
    embedding_doc.initialized_value()
    # sess.run(init)
    for i in range(10000):
        x_1, x_2, _y = get_input()
        _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        if i % 50 == 0:
            print(i, " loss ", _loss)
    embed_doc = embedding_doc.eval(sess)
    np.save("../temp/embedding_test_doc.npy", embed_doc)

train()
# model_eval()
