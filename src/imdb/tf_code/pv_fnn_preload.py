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
HIDDEN_SIZE = 500


def fnn_model():
    x=np.load('../../../temp/imdb/tf_code/pv_pretrain/docembed.npy')
    x_train = x[:25000]
    x_test = x[25000:]
    y_train = np.zeros((25000,), dtype=np.int64)
    y_test = np.zeros((25000,), dtype=np.int64)
    y_train[12500:25000] = np.ones((12500,), dtype=np.int64)
    y_test[12500:25000] = np.ones((12500,), dtype=np.int64)

    feature_columns=[tf.feature_column.numeric_column('x',shape=[HIDDEN_SIZE])]
    classifier=tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[],
                                          n_classes=2)
                                          #model_dir=root_dir+'temp/imdb/tf_code/pv_pretrain')

    train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':x_train},
        y=y_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        batch_size=250,
        x={'x': x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False
    )
    classifier.train(input_fn=train_input_fn,steps=40000)
    accuracy_score=classifier.evaluate(input_fn=test_input_fn)['accuracy']
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


if __name__== '__main__':
    fnn_model()