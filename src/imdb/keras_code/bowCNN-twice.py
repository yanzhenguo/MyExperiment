# -*- coding: utf-8 -*-

'''use semi-supervised n-gram features as another input'''

import keras
import numpy as np
from keras.layers import Dense, GlobalMaxPooling1D, Input, Embedding, \
    AveragePooling1D, GlobalAveragePooling1D, Conv1D, Activation
from keras.models import Model, load_model
import get_data

num_words = 30000
max_len = 500

x_train, y_train, x_test, y_test, embedding_matrix= get_data.data_cnn(num_words, max_len, use_word2vec=True)

indice1 = np.arange(25000)
np.random.shuffle(indice1)
x_train = x_train[indice1]
y_train = y_train[indice1]

indice2 = np.arange(25000)
np.random.shuffle(indice2)
x_test = x_test[indice2]
y_test = y_test[indice2]

model1 = load_model('../temp/semi-bowCNN.model.h5')
embedding_matrix1 = model1.get_layer('embedding_1').get_weights()[0]

model2 = load_model('../temp/bowCNN.model.h5')
embedding_matrix2 = model2.get_layer('embedding_1').get_weights()[0]

main_input = Input(shape=(max_len,))
# model.add(Embedding(35000,50,input_length=500))
init_method = keras.initializers.Orthogonal()
# embedding1 = Embedding(num_words, 500, embeddings_initializer=init_method)(main_input)
embedding1 = Embedding(num_words, 500, weights=[embedding_matrix2], trainable=False)(main_input)

x = AveragePooling1D(pool_size=5, strides=1, padding='valid')(embedding1)
# x = GlobalMaxPooling1D()(x)
# x=GlobalAveragePooling1D()(x)
# x = Activation('relu')(x)

embedding2 = Embedding(num_words, 200, weights=[embedding_matrix1], trainable=False)(main_input)
y = AveragePooling1D(pool_size=5, strides=1, padding='valid')(embedding2)
y = Activation('relu')(y)
y = Conv1D(filters=500, kernel_size=1, padding='valid', use_bias=False, strides=1)(y)
# y = GlobalMaxPooling1D()(y)

# embedding3=Embedding(num_words,50,input_length=max_len,embeddings_initializer='normal')(input)
# p=GlobalAveragePooling1D()(embedding3)

# z=keras.layers.concatenate([x,y])
z = keras.layers.add([x, y])
# z = keras.layers.average([x, y])
z = Activation('relu')(z)
z = GlobalMaxPooling1D()(z)
# z=keras.layers.concatenate([x,y,p])
# x=Dropout(0.2)(x)
output = Dense(1, activation='sigmoid', use_bias=False)(z)

model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))

# predict = model.predict(x_test)
# out = open('../temp/prediction.txt', 'w')
# for i in range(25000):
#     if (y_test[i] == 0 and predict[i, 0] > 0.5) or (y_test[i] == 1 and predict[i, 0] < 0.5):
#         out.write(str(int(y_test[i])) + ' ' + str(predict[i, 0]) + '\n')
# out.close()
# model.summary()
# model.save('../temp/bowCNN.model.h5')
