# -*- coding: utf-8 -*-
'''
    使用无监督的方式训练n-gram embedding
'''
import pickle
from collections import Counter
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

batch_size = 32
num_words = 40000
max_len=300
data_dir = '../../../temp/imdb/tf_code/pv_ngram/'


def prepare_data():
    print("prepare training data")
    with open('../../../temp/imdb/keras_code/utils/texts.pkl', 'rb') as f:
        text1 = pickle.load(f)
    with open('../../../temp/imdb/keras_code/utils/texts_unsup.pkl', 'rb') as f:
        texts = text1[:25000]+pickle.load(f)

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
    sequences = tokenizer.texts_to_sequences(text1)
    sequences= pad_sequences(sequences, maxlen=max_len)
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


    x = np.zeros((50000, (max_len-2)*3), dtype=np.int32)
    for i,seq in enumerate(sequences):
        for j in range(len(seq)-2):
                x[i,j:j+3]=seq[j:j+3]

    x_train = x[:25000]
    x_test = x[25000:]
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

    return x_train,x_test,y_train,y_test, embedding_matrix


def get_input(x, y, start):
    sum_words = 25000
    if start + batch_size < sum_words:
        start += batch_size
        x_train_batch=x[start - batch_size:start]
        y_train_batch=y[start - batch_size:start]
        x_train_batch=np.reshape(x_train_batch,[batch_size*(max_len-2),3])
        return x_train_batch, y_train_batch, start

    else:
        # print('loop 2')
        # print(start, batch_size, sum_words)
        start = batch_size - sum_words + start
        x_train_batch =np.concatenate((x[start - batch_size:], x[:start]), axis=0)
        y_train_batch=np.concatenate((y[start - batch_size:], y[:start]), axis=0)
        x_train_batch = np.reshape(x_train_batch, [batch_size*(max_len-2), 3])
        return x_train_batch, y_train_batch, start


def train():
    x_place = tf.placeholder(dtype=tf.int64, shape=(None, 3))
    y_place = tf.placeholder(dtype=tf.int64, shape=(None,))
    with tf.device("/cpu:0"):
        embedding_word = tf.get_variable('embedding_word', [num_words, 300], trainable=False)
        input_1 = tf.nn.embedding_lookup(embedding_word, x_place)
    byte_list = tf.unstack(input_1, axis=1)
    with tf.variable_scope("myrnn"):
        cell = tf.contrib.rnn.GRUCell(300)
        output, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, dtype=tf.float32)
    encoding_2=tf.reshape(encoding,[-1,(max_len-2),300])
    byte_list_2 = tf.unstack(encoding_2, axis=1)
    with tf.variable_scope("doc_rnn"):
        cell_2 = tf.contrib.rnn.GRUCell(300)
        output_2, encoding_3 = tf.contrib.rnn.static_rnn(cell_2, byte_list_2, dtype=tf.float32)

    out1 = tf.layers.dense(encoding_3, 2, activation=None)
    output2 = tf.nn.softmax(out1)
    predicted_classes = tf.argmax(output2, 1)

    a = tf.cast(tf.equal(y_place, predicted_classes), tf.float32)
    accuracy = tf.reduce_mean(a)
    onehot_labels = tf.one_hot(y_place, 2, 1, 0)
    print(onehot_labels)
    loss = tf.losses.mean_squared_error(onehot_labels, output2)

    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()

    val_to_save = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="myrnn")
    print(val_to_save)
    saver = tf.train.Saver(val_to_save)

    sess = tf.Session()
    sess.run(init)
    x_train, x_test, y_train, y_test, embedding_matrix= prepare_data()
    init_nce = tf.assign(embedding_word, embedding_matrix)
    sess.run(init_nce)
    saver.restore(sess, '../../../temp/imdb/tf_code/pv_ngram/pv_ngram.ckpt')

    start = 0
    for i in range(10000):
        x_1, _y, start = get_input(x_train, y_train, start)
        # _loss, _ = sess.run([loss, optimizer], feed_dict={x1_place: x_1, x2_place: x_2, y_place: _y})
        _acc, _ = sess.run([accuracy, optimizer], feed_dict={x_place: x_1, y_place: _y})
        if i % 300 == 0:
            print(i, " accuracy is: ", _acc)
        if i%500==0:
            sum_acc = 0
            sum_loss = 0
            for j in range(100):
                x_test_batch= np.reshape(x_test[250*j:250*(j+1)], [250 * (max_len - 2), 3])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x_place: x_test_batch,
                                                        y_place: y_test[250*j:250*(j+1)]})
                sum_acc += val_acc
                sum_loss += val_loss
            print("val loss: ", sum_loss / 100, "val acc: ", sum_acc / 100)


if __name__ == "__main__":
    train()
