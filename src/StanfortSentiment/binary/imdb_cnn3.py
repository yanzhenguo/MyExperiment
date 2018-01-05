import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical


maxlen = 50
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 30
X_train = np.load('Xtrain.npy')
X_test = np.load('Xtest.npy')
y_train = np.load('Ytrain.npy')
y_test = np.load('Ytest.npy')
embedding_matric = np.load('embeddingMatrix.npy')

#indice = np.arange(117209)
#np.random.shuffle(indice)
#X_train=X_train[indice]
#y_train=y_train[indice]


print('Build model...')
model = Sequential()
model.add(Embedding(16190,embedding_dims,input_length=maxlen,weights=[embedding_matric],trainable=False))
model.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu'))
model.add(GlobalMaxPooling1D(name='pool'))
model.add(Dense(hidden_dims,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,validation_data=(X_test, y_test))

# tmodel = Model(input=model.input,output=model.get_layer('pool').output)
# train =tmodel.predict(X_train)
# test = tmodel.predict(X_test)
# np.save('XtrainCov.npy',train)
# np.save('XtestCov.npy',test)
