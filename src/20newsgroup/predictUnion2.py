#combine cnn and skip-thought
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D,Merge
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical

max_features = 20000
maxlen = 1000
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 20

basedir = '/media/yan/My Passport/news20/FirstMethod/'
X_train = np.load(basedir+'convXtrain.npy')
X_test = np.load(basedir+'convXtest.npy')
embedding_matric = np.load(basedir+'embeddingMatrix.npy')
model2 = Sequential()
model2.add(Embedding(179174,embedding_dims,weights=[embedding_matric],input_length=maxlen,dropout=0.2,trainable=False))
model2.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1))
model2.add(GlobalMaxPooling1D(name='pool'))
model2.add(Dense(250,activation='relu'))


Xtrain = np.load('/media/yan/My Passport/news20/SecondMethod/Xtrain.npy')
Ytrain = np.load('/media/yan/My Passport/news20/SecondMethod/Ytrain.npy')
Xtest = np.load('/media/yan/My Passport/news20/SecondMethod/Xtest.npy')
Ytest = np.load('/media/yan/My Passport/news20/SecondMethod/Ytest.npy')
Ytrain=to_categorical(Ytrain)
Ytest=to_categorical(Ytest)
model1 = Sequential()
model1.add(Dense(2400,input_dim=4800,activation='relu'))
#model1.add(Dropout(0.8))
model1.add(Dense(250,activation='relu'))


merged = Merge([model1, model2], mode='concat')
model3 = Sequential()
model3.add(merged)
#model3.add(Dense(120,activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(20,activation='softmax'))
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc']) #0.8588
model3.fit([Xtrain,X_train], Ytrain,validation_data=([Xtest,X_test],Ytest),nb_epoch=30, batch_size=32)
