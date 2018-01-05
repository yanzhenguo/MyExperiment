'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D,MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical

max_features = 30000
maxlen = 100
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 30
basedir = '/media/yan/My Passport/StanfordSentiment/FourthMethod/'
X_train = np.load(basedir+'Xtrain.npy')
X_test = np.load(basedir+'Xtest.npy')
Y = np.load(basedir+'Yall.npy')
y_train=Y[0:7077]
y_test=Y[7077:8826]
#y_test=Y[8826:9711]
embedding_matric = np.load(basedir+'embeddingMatrix.npy')



print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(16317,embedding_dims,weights=[embedding_matric],input_length=maxlen,trainable=False))

model.add(Convolution1D(128,5,activation='relu'))
model.add(MaxPooling1D(5))
model.add(Convolution1D(128,5,activation='relu'))
model.add(MaxPooling1D(5))
model.add(Convolution1D(128,5,activation='relu'))
model.add(MaxPooling1D(35))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

tmodel = Model(input=model.input,output=model.get_layer('pool').output)
train =tmodel.predict(X_train)
test = tmodel.predict(X_test)
np.save(basedir+'XtrainCov.npy',train)
np.save(basedir+'XtestCov.npy',test)
