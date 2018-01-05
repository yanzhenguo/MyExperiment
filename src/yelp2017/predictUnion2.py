from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D,Merge
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical

maxlen = 1000
batch_size = 64
embedding_dims = 300
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 20

basedir = '/home/yan/Yelp2017/FirstMethod/'
X = np.load(basedir+'Xall.npy')
Y = np.load(basedir+'label.npy')
lenX = 200000
nb_trainsample=int(0.8*lenX)
X_train = X[0:nb_trainsample]
y_train = Y[0:nb_trainsample]
X_test = X[nb_trainsample:lenX]
y_test = Y[nb_trainsample:lenX]

#indice = np.arange(nb_trainsample)
#np.random.shuffle(indice)
#X_train=X_train[indice]
#y_train=y_train[indice]
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
embedding_matric = np.load(basedir+'embeddingMatrix.npy')

model2 = Sequential()
model2.add(Embedding(192318,embedding_dims,weights=[embedding_matric],input_length=maxlen,trainable=True))
model2.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1))
model2.add(GlobalMaxPooling1D(name='pool'))
model2.add(Dense(250,activation='relu'))
#model2.add(Dropout(0.5))


basedir = '/home/yan/Yelp2017/SecondMethod/'
X = np.load(basedir+'DocVector.npy')
Y = np.load(basedir+'label.npy')
Xtrain = X[0:160000]
Ytrain = Y[0:160000]
Xtest = X[160000:200000]
Ytest = Y[160000:200000]
#indice = np.arange(40000)
#np.random.shuffle(indice)
#Xtrain = Xtrain[indice]
#Ytrain = Ytrain[indice]
Ytrain = to_categorical(Ytrain)
Ytest = to_categorical(Ytest)
model1 = Sequential()
model1.add(Dense(2400,input_dim=4800,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(250,activation='relu'))


merged = Merge([model1, model2], mode='concat')
model3 = Sequential()
model3.add(merged)
model3.add(Dense(250,activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(5,activation='softmax'))
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model3.fit([Xtrain,X_train], Ytrain,validation_data=([Xtest,X_test],Ytest),nb_epoch=30, batch_size=32)
