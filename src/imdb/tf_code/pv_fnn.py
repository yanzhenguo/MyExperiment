'''
将pv_prtrain训练得到的模型得到文档的向量，然后使用全连接网络进行分类
'''
import pickle
import math
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

root_dir='../../../'
batch_size=50
num_train=25000
HIDDEN_SIZE = 500


start = 0


def get_input(x_train,y_train):
    global start
    if start+batch_size<num_train:
        x = x_train[start:start+batch_size]
        y = y_train[start:start+batch_size]
        start+=batch_size
    else:
         x =np.concatenate((x_train[start:num_train],x_train[:num_train-start]), axis=0)
         y= np.concatenate((y_train[start:num_train],y_train[:num_train-start]))
         start = num_train-start
    return x,y


def fnn_model():
    x=np.load('../../../temp/imdb/tf_code/pv_pretrain/docembed.npy')
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

    x_place = tf.placeholder(dtype=tf.float32, shape=(None, HIDDEN_SIZE))
    y_place = tf.placeholder(dtype=tf.int64, shape=(None,))
    # out1 = tf.layers.dense(x_place, 300, activation=None)
    # out2=tf.nn.relu(out1)
    out3=tf.layers.dense(x_place,2,activation=None)
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

    for i in range(10000):
        x,y=get_input(x_train,y_train)
        _loss, _acc, _ = sess.run([loss, accuracy, train_op],
                                  feed_dict={x_place: x, y_place: y})
        if i % 100 == 0:
            print("iter: %d loss: %f accuracy: %f" % (i, _loss, _acc), )
        if i % 500 == 0:
            sum_acc = 0
            sum_loss = 0
            for j in range(25):
                _val_loss, _val_acc = sess.run([loss, accuracy], feed_dict={x_place: x_test[1000 * j:1000 * (j + 1)],
                                                                            y_place: y_test[1000 * j:1000 * (j + 1)]})
                sum_acc += _val_acc
                sum_loss += _val_loss
            print('val acc:', sum_acc / 25, 'val loss: ', _val_loss / 25)
    sess.close()


if __name__== '__main__':
    fnn_model()