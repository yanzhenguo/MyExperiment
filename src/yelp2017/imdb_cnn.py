import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Embedding,Convolution1D, GlobalMaxPooling1D,Conv1D
from keras.utils.np_utils import to_categorical


maxlen = 1000
batch_size = 64
embedding_dims = 300
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 20

X = np.load('Xall.npy')
Y = np.load('label.npy')
lenX = X.shape[0]
nb_trainsample=int(0.8*lenX)
X_train = X[0:nb_trainsample]
y_train = Y[0:nb_trainsample]
X_test = X[nb_trainsample:]
y_test = Y[nb_trainsample:lenX]

#indice = np.arange(y_train.shape[0])
#np.random.shuffle(indice)
#X_train=X_train[indice]
#y_train=y_train[indice]
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
embedding_matric = np.load('embeddingMatrix.npy')



print('Build model...')
model = Sequential()
model.add(Embedding(192318,embedding_dims,weights=[embedding_matric],input_length=maxlen,trainable=False,dropout=0.1,name='layer1'))
model.add(Conv1D(filters=250,kernel_size=3, padding='valid',activation='relu'))
model.add(GlobalMaxPooling1D(name='layer3'))
model.add(Dense(hidden_dims,activation='relu',name='layer4'))
model.add(Dropout(0.3))
model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,validation_data=(X_test, y_test))

#tmodel = Model(input=model.input,output=model.get_layer('layer3').output)
#train =tmodel.predict(X_train)
#test = tmodel.predict(X_test)
#np.save(basedir+'XtrainCov.npy',train)
#np.save(basedir+'XtestCov.npy',test)
