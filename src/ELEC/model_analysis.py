import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D

model=load_model('../temp/seqCNN_3_model.h5')
print(model.to_json())
embedding = model.get_layer('embedding_1')
weights = embedding.get_weights()[0]
dense = model.get_layer('dense_1')
weight_dense = dense.get_weights()[0]
print(weight_dense)
#print(weight[1,:])

# Xtrain = np.load('../temp/Xtrain_seqcnn.npy')
# Xtest = np.load('../temp/Xtest_seqcnn.npy')
#
# f = codecs.open('../temp/words.pkl','rb')
# words = pickle.load(f)
# f.close()
#
# average_weights = np.zeros((800,598))
# for i in range(598):
#     average_weights[:,i]=weight[Xtrain[0,i*3]]+weight[Xtrain[0,i*3+1]]+weight[Xtrain[0,i*3+2]]
# for i in range(800):
#     temp = average_weights[i]
#     j = np.where(temp == np.max(temp))[0][0]
#     id1 = int(Xtrain[0, j*3])
#     id2 = int(Xtrain[0, j * 3+1])-30000
#     id3 = int(Xtrain[0, j * 3+2]) - 60000
#     print(words[id1]+' '+words[id2] + ' ' + words[id3])

