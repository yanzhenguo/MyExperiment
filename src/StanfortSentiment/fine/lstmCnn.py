from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Embedding,Convolution1D,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical


maxlen = 50
batch_size = 64
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 30
basedir = '/media/yan/My Passport/StanfordSentiment/FirstMethod/'
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



print('Build model...')
model = Sequential()
model.add(Embedding(17838,embedding_dims, input_length=maxlen,weights=[embedding_matric],trainable=False))
model.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu'))
#model.add(MaxPooling1D(2,name='pool'))
model.add(LSTM(250))
model.add(Dense(120,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test, y_test))

#tmodel = Model(input=model.input,output=model.get_layer('pool').output)
#train =tmodel.predict(X_train)
#test = tmodel.predict(X_test)
#np.save(basedir+'XtrainCov.npy',train)
#np.save(basedir+'XtestCov.npy',test)
