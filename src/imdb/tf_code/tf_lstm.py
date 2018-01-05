# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
sys.path.append('../')
from keras_code import get_data

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 300
MAX_LABEL = 2
batch_size = 32
[x_train, y_train, x_test, y_test] = get_data.data_cnn(30000, 300,use_word2vec=False)


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


def main():

  x_place=tf.placeholder(dtype=tf.int32,shape=(None,300))
  y_place=tf.placeholder(dtype=tf.int64,shape=(None,))

  with tf.device("/cpu:0"):
      embedding = tf.Variable(tf.random_uniform([30000, 300], -0.5, 0.5))
      inputs = tf.nn.embedding_lookup(embedding, x_place)
  byte_list = tf.unstack(inputs, axis=1)

  cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  output = tf.nn.softmax(logits)
  predicted_classes = tf.argmax(logits, 1)
  a = tf.cast(tf.equal(y_place, predicted_classes),tf.float32)
  accuracy = tf.reduce_mean(a)
  onehot_labels = tf.one_hot(y_place, MAX_LABEL, 1, 0)
  loss = tf.losses.mean_squared_error(onehot_labels, output)

  train_op = tf.train.AdamOptimizer().minimize(loss)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  for i in range(20000):
      x, y = get_input()
      _loss, _acc, _ = sess.run([loss,accuracy, train_op],{x_place:x,y_place:y})
      if i%100==0:
          print("iter: %d loss: %f accuracy: %f" % (i, _loss, _acc))
      if i%100==0:
          sum_acc=0
          for j in range(25):
            _, acc = sess.run([loss,accuracy],feed_dict={x_place: x_test[1000*j:1000*(j+1)],
                                                       y_place: y_test[1000*j:1000*(j+1)]})
            sum_acc+=acc
          print('test acc:',sum_acc/25)

if __name__ == '__main__':
  main()
