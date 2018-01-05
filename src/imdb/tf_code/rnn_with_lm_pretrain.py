# -*- coding: utf-8 -*-
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
import codecs
import tensorflow as tf


MAX_DOCUMENT_LENGTH = 400
vocab_size = 30000
num_train = 23000
Embed_dimension = 400
HIDDEN_SIZE = 500
batch_size = 32

embedding_matrix = None;

def prepare_data():
    print("prepare training data")
    global embedding_matrix;
    f = codecs.open('../../../temp/imdb/keras_code/utils/texts.pkl', 'rb')
    text = pickle.load(f)
    f.close()
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    sequence_pad = pad_sequences(sequence, maxlen=MAX_DOCUMENT_LENGTH, dtype=np.int32, padding='post',
                                 truncating='post')
    # prepare pretrained word2vec vectors
    # word_index = tokenizer.word_index
    # embeddings_index = {}
    # wordX = np.load('/media/yan/winD/docker/word2vec/words/word2vec.npy')
    # f = codecs.open('/media/yan/winD/docker/word2vec/words/wordsInWord2vec.pkl', 'rb')
    # allwords = pickle.load(f)
    # f.close()
    # for i in range(3000000):
    #     embeddings_index[''.join(allwords[i])] = wordX[i, :]
    # embedding_matrix = np.zeros((vocab_size, 300))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None and i < vocab_size:
    #         embedding_matrix[i] = embedding_vector

    seq_len = []
    for i in range(len(sequence)):
        r = len(sequence[i])
        if r < MAX_DOCUMENT_LENGTH:
            seq_len.append(r)
        else:
            seq_len.append(MAX_DOCUMENT_LENGTH)

    x_train = sequence_pad[:25000]
    x_test = sequence_pad[25000:]
    x_train_len = np.asarray(seq_len[:25000])
    x_test_len = np.asarray(seq_len[25000:])
    y_train = np.zeros((25000,), dtype=np.int8)
    y_test = np.zeros((25000,), dtype=np.int8)
    y_train[12500:25000] = np.ones((12500,), dtype=np.int8)
    y_test[12500:25000] = np.ones((12500,), dtype=np.int8)

    indice = np.arange(25000)
    np.random.shuffle(indice)
    x_train = x_train[indice]
    x_test = x_test[indice]
    y_train = y_train[indice]
    y_test = y_test[indice]
    x_train_len = x_train_len[indice]
    x_test_len = x_test_len[indice]

    return x_train,x_test,y_train,y_test, x_train_len,x_test_len


x_train, x_test, y_train, y_test, x_train_len, x_test_len = prepare_data()


start = 0
def get_input():
    global start

    num_all = num_train
    if start + batch_size < num_all:
        x_place = x_train[start:start + batch_size]
        x_len = x_train_len[start:start + batch_size]
        yp = y_train[start:start + batch_size]
        start += batch_size
    else:
        x_place = np.concatenate((x_train[start:num_all], x_train[:start - num_all + batch_size]), axis=0)
        x_len = np.concatenate((x_train_len[start:num_all], x_train_len[:start - num_all + batch_size]))
        yp = np.concatenate((y_train[start:num_all], y_train[:start - num_all + batch_size]))
        start = start - num_all + batch_size
    return x_place, x_len, yp


def train():
    x_place = tf.placeholder(dtype=tf.int64, shape=(None, MAX_DOCUMENT_LENGTH))
    x_len_place = tf.placeholder(dtype=tf.int64, shape=(None,))
    y_place = tf.placeholder(dtype=tf.int64, shape=(None,))
    with tf.device("/cpu:0"):
        embedding_word = tf.Variable(tf.random_uniform([vocab_size, Embed_dimension], -0.5, 0.5))
        input = tf.nn.embedding_lookup(embedding_word, x_place)
    # input_drop = tf.nn.dropout(input,0.5)
    byte_list = tf.unstack(input, axis=1)
    with tf.variable_scope("myrnn"):
        cell = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, sequence_length=x_len_place, dtype=tf.float32)
        # output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, dtype=tf.float32)
    with tf.variable_scope("mydense"):
        dense1 = tf.layers.dense(encoding,2)
    logits = tf.nn.softmax(dense1)

    predicted_classes = tf.argmax(logits, 1)
    a = tf.cast(tf.equal(y_place, predicted_classes), tf.float32)
    accuracy = tf.reduce_mean(a)
    onehot_labels = tf.one_hot(y_place, 2, 1, 0)
    loss = tf.losses.mean_squared_error(onehot_labels, logits)


    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    val_to_save = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="myrnn")
    val_to_save.append(embedding_word)
    saver = tf.train.Saver(val_to_save)
    # assign = embedding_word.assign(embedding_matrix)
    with tf.Session() as sess:
        sess.run(init)
        # sess.run(assign)
        saver.restore(sess, '../../../temp/imdb/tf_code/lm_pretrain/lm_pretrain.ckpt')

        for i in range(10000):
            x_1, x_len, _y = get_input()
            # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
            _loss, _acc, _ = sess.run([loss, accuracy, optimizer], feed_dict={x_place: x_1,x_len_place:x_len, y_place: _y})
            if i % 300 == 0:
                print(i, " loss ", _loss,"acc ", _acc)
            if i%300==0:
                sum_acc=0
                sum_loss=0
                for j in range(23000,25000,50):
                    val_loss,val_acc = sess.run([loss, accuracy],
                                                feed_dict={x_place: x_train[j:j+50], x_len_place:x_train_len[j:j+50],
                                                y_place: y_train[j:j+50]})
                    sum_acc+=val_acc
                    sum_loss+=val_loss
                print("val loss: ", sum_loss/40, "val acc: ", sum_acc/40)
            if i%300==0:
                sum_acc=0
                sum_loss=0
                for j in range(250):
                    test_loss, test_acc = sess.run([loss, accuracy],
                                                feed_dict={x_place: x_test[j:j+100], x_len_place:x_test_len[j:j+100],
                                                y_place: y_test[j:j+100]})
                    sum_acc+=test_acc
                    sum_loss+=test_loss
                print("test loss: ", sum_loss/250, "test acc: ", sum_acc/250)


train()
