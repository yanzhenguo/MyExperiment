# from keras.datasets import imdb
from urllib.request import urlopen

import numpy as np
import csv
import codecs
import pickle
import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, TimeDistributed,Dense,Conv1D

root_dir = '../../'
y = np.load(root_dir + 'temp/yelp2015_p/extract_data/label.npy')
print(y.shape)
# nngn_predict=np.zeros([25000],np.int)
# with open('/home/yan/bow-NNGN_predction.txt') as f:
#     for i,line in enumerate(f.readlines()):
#         nngn_predict[i]=int(line.strip('\n'))
#
# rnn_nngn_predict=np.zeros([25000],np.int)
# with open('/home/yan/rnn_nngn_predict') as f:
#     for i, line in enumerate(f.readlines()):
#         rnn_nngn_predict[i]=int(line.strip('\n'))
# num=0
# for i in range(25000):
#     if i<12500 and nngn_predict[i]==1 and rnn_nngn_predict[i]==0:
#         print(i)
#         num+=1
#     elif i>=12500 and nngn_predict[i]==0 and rnn_nngn_predict[i]==1:
#         print(i)
#         num+=1
# print(num)