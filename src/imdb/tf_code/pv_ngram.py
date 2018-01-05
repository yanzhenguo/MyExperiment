# -*- coding: utf-8 -*-
'''
    使用无监督的方式训练n-gram embedding
'''
import pickle
from collections import Counter
import numpy as np
from keras.preprocessing.text import Tokenizer
import tensorflow as tf

batch_size = 256
num_sampled = 256
num_words = 40000
num_ngram = None
data_dir = '../../../temp/imdb/tf_code/pv_ngram/'


def tokenize(sentence):
    words = sentence.split(' ')
    # tokens = []
    # for i in range(len(words) - 2):
    #     tokens += ["_*_".join(words[i:i + 3])]
    # return tokens
    tokens = []
    for i in range(len(words) - 2):
        tokens.append(tuple(words[i:i + 3]))
    return tokens


def prepare_data():
    global num_ngram
    print("prepare training data")
    with open('../../../temp/imdb/keras_code/utils/texts.pkl', 'rb') as f:
        texts = pickle.load(f)[:25000]
    with open('../../../temp/imdb/keras_code/utils/texts_unsup.pkl', 'rb') as f:
        texts += pickle.load(f)

    with open('/home/yan/my_datasets/glove/words.pkl', 'rb') as f:
        glove_words = pickle.load(f)
        glove_words_set = set(glove_words)
    for i, s in enumerate(texts):
        new_s = []
        for token in s.split(' '):
            if token in glove_words_set:
                new_s.append(token)
        texts[i] = ' '.join(new_s)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    embeddings_index = {}
    wordX = np.load('/home/yan/my_datasets/glove/embedding.300d.npy')
    for i in range(len(glove_words)):
        embeddings_index[glove_words[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    all_ngram = set()
    for seq in sequences:
        for i in range(len(seq) - 2):
            all_ngram.add(tuple(seq[i:i + 3]))
    num_ngram = len(all_ngram)
    all_ngram=list(all_ngram)
    print('there are %d ngrams' % (num_ngram))

    x = np.zeros((num_ngram * 3, 3), dtype=np.int32)
    y = np.zeros((num_ngram * 3, 1), dtype=np.int32)
    index = 0
    for i, seq in enumerate(all_ngram):
        for id in seq:
            x[index] = seq
            y[index] = id
            index += 1
    indice = np.arange(num_ngram * 3)
    np.random.shuffle(indice)
    x = x[indice]
    y = y[indice]

    print(x.shape,y.shape)

    return x, y, embedding_matrix


def get_input(x, y, start):
    sum_words = num_ngram * 3
    if start + batch_size < sum_words:
        start += batch_size
        return x[start - batch_size:start], y[start - batch_size:start], start

    else:
        # print('loop 2')
        # print(start, batch_size, sum_words)
        start = batch_size - sum_words + start
        return np.concatenate((x[start - batch_size:], x[:start]), axis=0), \
               np.concatenate((y[start - batch_size:], y[:start]), axis=0), start


def train():
    x_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 3))
    y_place = tf.placeholder(dtype=tf.int64, shape=(batch_size, 1))
    with tf.device("/cpu:0"):
        embedding_word = tf.get_variable('embedding_word', [num_words, 300], trainable=False)
        nce_weights = tf.get_variable('nce_weights_words', [num_words, 300], trainable=True)
        nce_biases = tf.Variable(tf.zeros([num_words]), trainable=True)

        input_1 = tf.nn.embedding_lookup(embedding_word, x_place)
    byte_list = tf.unstack(input_1, axis=1)
    with tf.variable_scope("myrnn"):
        cell = tf.contrib.rnn.GRUCell(300)
        output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, dtype=tf.float32)
    # print(output)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y_place,
                       inputs=encoding,
                       num_sampled=num_sampled,
                       num_classes=num_words))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()

    val_to_save = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="myrnn")
    saver = tf.train.Saver(val_to_save)

    sess = tf.Session()
    sess.run(init)
    x, y, embedding_metrix = prepare_data()
    init_nce = tf.assign(embedding_word, embedding_metrix)
    sess.run(init_nce)
    start = 0
    for i in range(100000):
        x_1, _y, start = get_input(x, y, start)
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _loss, _ = sess.run([loss, optimizer], feed_dict={x_place: x_1, y_place: _y})
        if i % 300 == 0:
            print(i, " loss ", _loss)

    saver.save(sess, data_dir+'pv_ngram.ckpt')


if __name__ == "__main__":
    train()
