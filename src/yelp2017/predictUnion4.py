import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Embedding,Convolution1D,GlobalMaxPooling1D,Merge,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical

maxlen = 50
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
basedir = '/home/yan/StanfordSentiment/FirstMethod/'
X_train = np.load(basedir+'Xtrain.npy')
X_test = np.load(basedir+'Xtest.npy')
y_train = np.load(basedir+'Ytrain.npy')
y_test = np.load(basedir+'Ytest.npy')

indice = np.arange(y_train.shape[0])
np.random.shuffle(indice)
X_train=X_train[indice]
y_train=y_train[indice]
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

embedding_matric = np.load(basedir+'embeddingMatrix.npy')

model2 = Sequential()
model2.add(Embedding(17838,embedding_dims,weights=[embedding_matric],input_length=maxlen,trainable=True))
model2.add(LSTM(250,return_sequences=True))
model2.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length,activation='relu'))
model2.add(GlobalMaxPooling1D())

model2.add(Dense(250,activation='relu'))
#model2.add(Dropout(0.3))


basedir = '/home/yan/StanfordSentiment/SecondMethod/'
X = np.load(basedir+'texts.npy')
Xtrain = X[0:8544]
Xtest = X[8544:10754]
Ytrain = np.load(basedir+'Ytrain.npy')
Ytest = np.load(basedir+'Ytest.npy')
Xtrain=Xtrain[indice]
Ytrain=Ytrain[indice]
Ytrain = to_categorical(Ytrain)
Ytest = to_categorical(Ytest)
model1 = Sequential()
model1.add(Dense(1200,input_dim=4800,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(250,activation='relu'))
#model1.add(Dense(50,activation='relu'))


merged = Merge([model1, model2], mode='concat')
model3 = Sequential()
model3.add(merged)
model3.add(Dense(250,activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(5,activation='softmax'))
model3.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['acc']) #0.8588
model3.fit([Xtrain,X_train], Ytrain,validation_data=([Xtest,X_test],Ytest),nb_epoch=30, batch_size=32)
