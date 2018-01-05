# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
from keras_code import get_data

vocabulary_size = 30000
embedding_dimension = 800
max_document_length = 1000
batch_size = 32

[x_train, y_train, x_test, y_test] = get_data.data_cnn(vocabulary_size, max_document_length, use_word2vec=False)

start = 0
def get_input():
    global  start
    if start+batch_size<25000:
        x = x_train[start:start+batch_size]
        y = y_train[start:start+batch_size]
        start+=batch_size
    else:
         x =np.concatenate((x_train[start:],x_train[:25000-start]), axis=0)
         y= np.concatenate((y_train[start:],y_train[:25000-start]))
         start = 25000-start
    return x,y


x_place=tf.placeholder(dtype=tf.int32,shape=(None,max_document_length))
y_place=tf.placeholder(dtype=tf.int64,shape=(None,))

with tf.device("/cpu:0"):
    embedding = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension], -0.5, 0.5))
    # embedding = tf.get_variable("embedding", [30000, HIDDEN_SIZE], dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, x_place)
re_input = tf.reshape(inputs,[-1, max_document_length, embedding_dimension,1])
mean_pool = tf.contrib.layers.avg_pool2d(re_input, [3, 1], [1, 1] )
max_pool = tf.contrib.layers.max_pool2d(mean_pool, [998,1], [1,1])
encoding = tf.reshape(max_pool, [-1, embedding_dimension])
logits = tf.layers.dense(encoding, 2, activation=None)
output = tf.nn.softmax(logits)
predicted_classes = tf.argmax(logits, 1)
a = tf.cast(tf.equal(y_place, predicted_classes),tf.float32)
accuracy = tf.reduce_mean(a)
onehot_labels = tf.one_hot(y_place, 2, 1, 0)
loss = tf.losses.mean_squared_error(onehot_labels, output)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(50000):
    x, y = get_input()
    _loss, _acc, _ = sess.run([loss, accuracy, train_op], {x_place: x, y_place: y})
    if i % 200 == 0:
        print("iter: %d loss: %f accuracy: %f" % (i, _loss, _acc))
    if i % 1000 == 0:
        sum_acc = 0
        for j in range(250):
            _, acc = sess.run([loss, accuracy], feed_dict={x_place: x_test[100 * j:100 * (j + 1)],
                                                           y_place: y_test[100 * j:100 * (j + 1)]})
            sum_acc += acc
        print('test acc:', sum_acc / 250)
