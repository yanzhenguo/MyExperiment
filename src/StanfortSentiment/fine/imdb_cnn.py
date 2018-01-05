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
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
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

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(17838,
                    embedding_dims,                  
                    input_length=maxlen,
                    weights=[embedding_matric],trainable=False
                    ))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
model.add(GlobalMaxPooling1D(name='pool'))
model.add(Dropout(0.5))
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
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
