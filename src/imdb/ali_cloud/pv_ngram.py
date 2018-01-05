# -*- coding: utf-8 -*-
'''
    使用无监督的方式训练n-gram embedding
'''
import pickle
import argparse
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO

batch_size=256
num_sampled=256
num_words=40000
num_ngram=400000
FLAGS=None


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
    global tokenizer
    print("prepare training data")
    with FileIO(os.path.join(FLAGS.buckets, 'imdb/texts.pkl'), 'r+') as f:
        texts = pickle.load(f)[:25000]
    with FileIO(os.path.join(FLAGS.buckets, 'imdb/texts_unsup.pkl'), 'r+') as f:
        texts += pickle.load(f)

    with FileIO(os.path.join(FLAGS.buckets, 'glove/words.pkl'), 'r+') as f:
        glove_words = pickle.load(f)
        glove_words_set=set(glove_words)
    for i,s in enumerate(texts):
        new_s=[]
        for token in s.split(' '):
            if token in glove_words_set:
                new_s.append(token)
        texts[i]=' '.join(new_s)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters=''
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index=tokenizer.word_index

    embeddings_index = {}
    wordX = np.load(FileIO(os.path.join(FLAGS.buckets, "glove/embedding.300d.npy"),mode='r+'))
    for i in range(len(glove_words)):
        embeddings_index[glove_words[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    ngram_text=[]
    for seq in sequences:
        s=[str(i) for i in seq]
        ngram_sentence=[]
        for i in range(len(seq)-2):
            ngram_sentence.append('_'.join(s[i:i+3]))
        ngram_text.append(' '.join(ngram_sentence))
    tokenizer2=Tokenizer(num_words=num_ngram)
    tokenizer2.filters=''
    tokenizer2.fit_on_texts(ngram_text)
    ngram_index=tokenizer2.word_index
    frequent_ngram=sorted(ngram_index.items(),key=lambda k:k[1])[:num_ngram]
    print('there are %d ngrams'%(len(ngram_index)))

    x = np.zeros((num_ngram*3, 1), dtype=np.int32)
    y = np.zeros((num_ngram*3, 1), dtype=np.int32)
    index = 0
    for i, seq in enumerate(frequent_ngram):
        word_ids=seq[0].split('_')
        for id in word_ids:
            x[index] = i
            y[index] = int(id)
            index += 1
    indice = np.arange(num_ngram*3)
    np.random.shuffle(indice)
    x = x[indice]
    y = y[indice]

    pickle.dump(word_index,FileIO(os.path.join(FLAGS.buckets, "word_index.pkl"),mode='w+'))
    pickle.dump(ngram_index,FileIO(os.path.join(FLAGS.buckets, "ngram_index.pkl"),mode='w+'))
    return x,y,embedding_matrix


def get_input(x, y, start):
    sum_words=num_ngram*3
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
        embedding_doc = tf.Variable(tf.random_uniform([num_ngram, 300], -0.5, 0.5))
        nce_weights = tf.get_variable('nce_weights_words',[num_words,300],trainable=True)
        nce_biases = tf.Variable(tf.zeros([num_words]),trainable=True)

        input_1 = tf.nn.embedding_lookup(embedding_doc, x_place)
    input_2 = tf.reshape(input_1, [-1, 300])
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y_place,
                       inputs=input_2,
                       num_sampled=num_sampled,
                       num_classes=num_words))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    x, y, embedding_metrix = prepare_data()
    init_nce=tf.assign(nce_weights,embedding_metrix)
    sess.run(init_nce)
    start = 0
    for i in range(1000000):
        x_1, _y, start = get_input(x, y, start)
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _loss, _ = sess.run([loss, optimizer], feed_dict={x_place: x_1, y_place: _y})
        if i % 300 == 0:
            print(i, " loss ", _loss)
    np.save(FileIO(os.path.join(FLAGS.buckets, "ngram_embedding.npy"),mode='w+'),embedding_doc.eval(sess))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    train()