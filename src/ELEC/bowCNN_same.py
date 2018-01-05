import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model , load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D, SpatialDropout1D, BatchNormalization

# in this script, different n-grams share the same word embedding
num_train = 25000
num_test = 25000
num_words = 20000
max_len = 500
f = codecs.open('../temp/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.filters = ''
tokenizer.fit_on_texts(texts[:num_train])
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("there are %d words" % (len(word_index)))
# sequences = []
# for i in range(num_train + num_test):
#     t = []
#     tokens = texts[i].split(' ')
#     for j in range(len(tokens)):
#         index = word_index.get(tokens[j], 0)
#         if index < num_words:
#             t.append(index)
#         else:
#             t.append(0)
#     sequences.append(t)

Xtrain = pad_sequences(sequences[:num_train], maxlen=max_len)
Xtest = pad_sequences(sequences[num_train:], maxlen=max_len)
Ytrain = np.zeros((num_train,), dtype=np.int8)
Ytest = np.zeros((num_test,), dtype=np.int8)
Ytrain[int(num_train / 2):] = np.ones((int(num_train / 2),), dtype=np.int8)
Ytest[int(num_test / 2):] = np.ones((int(num_test / 2),), dtype=np.int8)

indice1 = np.arange(num_train)
np.random.shuffle(indice1)
Xtrain = Xtrain[indice1]
Ytrain = Ytrain[indice1]

indice2 = np.arange(num_test)
np.random.shuffle(indice2)
Xtest = Xtest[indice2]
Ytest = Ytest[indice2]

# np.save('Xtrain',Xtrain)
# np.save('Xtest',Xtest)
# np.save('Ytrain',Ytrain)
# np.save('Ytest',Ytest)

# Xtrain=np.load('Xtrain.npy')
# Xtest=np.load('Xtest.npy')
# Ytrain=np.load('Ytrain.npy')
# Ytest=np.load('Ytest.npy')

input = Input(shape=(max_len,))
embedding = Embedding(num_words, 1000, embeddings_initializer=keras.initializers.Orthogonal())(input)
# x = SpatialDropout1D(rate=0.2)(embedding)
x = AveragePooling1D(pool_size=3, strides=1)(embedding)
# x = Activation('relu')(x)
# x = BatchNormalization(axis=1)(x)
x = GlobalMaxPooling1D()(x)
# x=GlobalAveragePooling1D()(x)

# y = AveragePooling1D(pool_size=2, strides=1)(embedding)
# y = GlobalMaxPooling1D()(y)
# y = GlobalAveragePooling1D()(y)

# p=GlobalAveragePooling1D()(embedding)
# p=GlobalMaxPooling1D()(embedding)

# z = keras.layers.concatenate([x,y])
# z = keras.layers.maximum([y,p])
# x = Dense(500,activation='relu')(x)
# x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# model = Model(inputs=input, outputs=output)
# model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
# model.fit(Xtrain, Ytrain, batch_size=32, epochs=20, validation_data=(Xtest, Ytest))
model=load_model('../temp/model.h5')
model.fit(Xtrain, Ytrain, batch_size=32, epochs=20, validation_data=(Xtest, Ytest))
