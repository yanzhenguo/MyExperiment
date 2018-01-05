# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

a=tf.constant([[1,2,3,4,5,6],[7,8,9,10,11,12]])
b=tf.reshape(a,[-1,3])
c=tf.reshape(a,[-1,6])


sess=tf.Session()
print(sess.run(c))