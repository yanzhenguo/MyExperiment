import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input,Embedding, \
    GlobalAveragePooling1D, AveragePooling1D

X_train = np.load('../temp/Xtrain_cnn.npy')
y_train = np.load('../temp/Ytrain.npy')
X_test = np.load('../temp/Xtest_cnn.npy')
y_test = np.load('../temp/Ytest.npy')
embedding_matric = np.load('../temp/embeddingMatrix.npy')


main_input = Input(shape=(500,))
x = Embedding(30000, 500, trainable=True)(main_input)
# x = Embedding(30000, 300, weights=[embedding_matric], trainable=False)(main_input)
x = Conv1D(filters=500, kernel_size=3, padding='valid', activation='relu')(x)
x = GlobalMaxPooling1D(name='pool')(x)
# x = Dense(250,activation='relu')(x)
# x = Dropout(0.2)(x)
# x = AveragePooling1D(pool_size=3, strides=1)(x)
# x = GlobalMaxPooling1D()(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[main_input], outputs=[output])
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
