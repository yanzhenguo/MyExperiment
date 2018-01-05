from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical



maxlen = 300
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 30
hidden_dims = 250
nb_epoch = 30
basedir = '/media/yan/My Passport/StanfordSentiment/FourthMethod/'
X_train = np.load(basedir+'Xtrain.npy')
X_test = np.load(basedir+'Xtest.npy')
Y = np.load(basedir+'Yall.npy')
y_train=Y[0:7077]
y_test=Y[7077:8826]
#y_test=Y[8826:9711]
indice = np.arange(7077)


print('Build model...')
model = Sequential()


model.add(Convolution1D(nb_filter=nb_filter,input_dim=74,input_length=300,
                        filter_length=filter_length,
                        activation='relu',))
# we use max pooling:
model.add(GlobalMaxPooling1D(name='pool'))

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

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
