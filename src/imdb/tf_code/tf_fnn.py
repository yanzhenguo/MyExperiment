# -*- coding: utf-8 -*-
'''将无监督得到的文档向量进行分类'''
import numpy as np
import tensorflow as tf

batch_size = 32
num_train = 23000
HIDDEN_SIZE = 500

train = np.load("../temp/embedding_train_doc.npy")
x_train= train[:25000]
x_test = np.load("../temp/embedding_test_doc.npy")
x_test = x_test[:25000]
y_train = np.zeros((25000,), dtype=np.int8)
y_test = np.zeros((25000,), dtype=np.int8)
y_train[12500:25000] = np.ones((12500,), dtype=np.int8)
y_test[12500:25000] = np.ones((12500,), dtype=np.int8)

indice = np.arange(25000)
np.random.shuffle(indice)
x_train = x_train[indice]
x_test = x_test[indice]
y_train = y_train[indice]
y_test  = y_test[indice]

# print(len(x_train)," ",len(y_train))

start = 0
def get_input():
    global  start
    if start+batch_size<num_train:
        x = x_train[start:start+batch_size]
        y = y_train[start:start+batch_size]
        start+=batch_size
    else:
         x =np.concatenate((x_train[start:num_train],x_train[:num_train-start]), axis=0)
         y= np.concatenate((y_train[start:num_train],y_train[:num_train-start]))
         start = num_train-start
    return x,y

x_place=tf.placeholder(dtype=tf.float32,shape=(None,HIDDEN_SIZE))
y_place=tf.placeholder(dtype=tf.int64,shape=(None,))
# out1 = tf.layers.dense(x_place, 200, activation=None)
# out2 = tf.nn.relu(out1)
out3 = tf.layers.dense(x_place, 2, activation=None)
output = tf.nn.softmax(out3)
predicted_classes = tf.argmax(output, 1)

a = tf.cast(tf.equal(y_place, predicted_classes), tf.float32)
accuracy = tf.reduce_mean(a)
onehot_labels = tf.one_hot(y_place, 2, 1, 0)
loss = tf.losses.mean_squared_error(onehot_labels, output)

train_op = tf.train.AdamOptimizer().minimize(loss)
# train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(20000):
    x, y = get_input()
    _loss, _acc, _ = sess.run([loss, accuracy, train_op], {x_place: x, y_place: y})
    if i % 500 == 0:
        print("iter: %d loss: %f accuracy: %f" % (i, _loss, _acc),)
    if i % 500 == 0:
        sum_acc = 0
        sum_loss =0
        for j in range(23,25):
            _val_loss, _val_acc = sess.run([loss, accuracy], feed_dict={x_place: x_train[1000 * j:1000 * (j + 1)],
                                                           y_place: y_train[1000 * j:1000 * (j + 1)]})
            sum_acc += _val_acc
            sum_loss += _val_loss
        print('val acc:', sum_acc/2, 'val loss: ',_val_loss/2)
    # if i % 500 == 0:
    #     sum_acc = 0
    #     sum_loss = 0
    #     for j in range(25):
    #         _test_loss, acc = sess.run([loss, accuracy], feed_dict={x_place: x_test[1000 * j:1000 * (j + 1)],
    #                                                        y_place: y_test[1000 * j:1000 * (j + 1)]})
    #         sum_acc += acc
    #         sum_loss += _test_loss
    #     print('test acc:', sum_acc / 25, 'test loss: ',sum_loss/25)