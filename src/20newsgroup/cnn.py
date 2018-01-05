import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Conv1D, GlobalMaxPooling1D,Input,Embedding,GlobalAveragePooling1D
#from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical



X_train = np.load('../temp/Xtrain_cnn.npy')
Y_train = np.load('../temp/Ytrain.npy')
X_test = np.load('../temp/Xtest_cnn.npy')
Y_test = np.load('../temp/Ytest.npy')
embedding_matric = np.load('../temp/embeddingMatrix.npy')
Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

# y_train = np.load('/media/yan/My Passport/FirstPaper/FourthMethod/convYtrain.npy')
# y_test = np.load('/media/yan/My Passport/FirstPaper/FourthMethod/convYtest.npy')

#indice = np.arange(25000)
#np.random.shuffle(indice)
#X_train = X_train[indice]
#y_train = y_train[indice]

model = Sequential()
#model.add(Embedding(35000,50,input_length=500))
model.add(Embedding(30000,300,weights=[embedding_matric],input_length=1000,trainable=False))
model.add(Conv1D(filters=1000,kernel_size=4,padding='valid',activation='relu'))
model.add(GlobalMaxPooling1D(name='pool'))
# model.add(Dense(250,activation='relu'))
# model.add(Dropout(0.2))
#model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.5))
model.add(Dense(20,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
model.fit(X_train, Y_train,batch_size=32,epochs=50,validation_data=(X_test, Y_test))