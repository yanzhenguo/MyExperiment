import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Embedding,Convolution1D, MaxPooling1D,GlobalMaxPooling1D,\
    LSTM,SpatialDropout1D,Bidirectional,Input, GRU


maxlen = 500
embedding_dims = 300
nb_filter = 250
filter_length = 3
hidden_dims = 250

X_train = np.load('../temp/Xtrain_cnn.npy')
y_train = np.load('../temp/Ytrain.npy')
X_test = np.load('../temp/Xtest_cnn.npy')
y_test = np.load('../temp/Ytest.npy')
embedding_matric = np.load('../temp/embeddingMatrix.npy')


print('Build model...')
input=Input(shape=(maxlen,))
x=Embedding(30000,embedding_dims,weights=[embedding_matric],input_length=maxlen,trainable=False)(input)
x=GRU(300, implementation=2)(x)
# model.add(Dense(120,activation='relu'))
# model.add(Dropout(0.2))
output=Dense(1,activation='sigmoid')(x)
model=Model(inputs=[input],outputs=[output])
model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=32,epochs=40,validation_data=(X_test, y_test))


