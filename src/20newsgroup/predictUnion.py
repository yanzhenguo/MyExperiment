#combine cnn and skip-thought
import codecs
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input,Merge,Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
Xtrain = np.load('/media/yan/My Passport/news20/SecondMethod/Xtrain.npy')
Ytrain = np.load('/media/yan/My Passport/news20/SecondMethod/Ytrain.npy')
Xtest = np.load('/media/yan/My Passport/news20/SecondMethod/Xtest.npy')
Ytest = np.load('/media/yan/My Passport/news20/SecondMethod/Ytest.npy')

Ytrain=to_categorical(Ytrain)
Ytest=to_categorical(Ytest)

print('begin to train ...')
model1 = Sequential()
model1.add(Dense(2400,input_dim=4800,activation='relu'))
#model1.add(Dropout(0.8))
model1.add(Dense(250,activation='relu'))
model1.add(Dropout(0.8))
model1.add(Dense(20,activation='relu'))


covXtrain = np.load('/media/yan/My Passport/news20/FirstMethod/XtrainCov.npy')
covXtest = np.load('/media/yan/My Passport/news20/FirstMethod/XtestCov.npy')
model2 = Sequential()
model2.add(Dense(250,activation='relu',input_dim=250))
model2.add(Dropout(0.8))
model2.add(Dense(20,activation='relu',name='model2_output'))


merged = Merge([model1, model2], mode='concat')

model3 = Sequential()
model3.add(merged)
#model3.add(Dropout(0.5))
model3.add(Dense(20,activation='softmax'))
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc']) #0.8588
model3.fit([Xtrain,covXtrain], Ytrain,validation_data=([Xtest,covXtest],Ytest),nb_epoch=30, batch_size=32)
