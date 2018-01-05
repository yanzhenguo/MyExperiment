import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D
# in this script, different n-grams share the same word embedding
num_words = 30000
max_len = 700
f = codecs.open('../temp/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[0:25000])
tokenizer.filters = ''
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# sequences = []
# for i in range(50000):
#     t = []
#     tokens = texts[i].lower().split(' ')
#     for j in range(len(tokens)):
#         index = word_index.get(tokens[j], 0)
#         if index < num_words:
#             t.append(index)
#         else:
#             t.append(0)
#     sequences.append(t)

sequences2 = []
for i in range(25000):
    a = sequences[i]
    sequences2.append(a)
    a.reverse()
    sequences2.append(a)

Xtrain = pad_sequences(sequences2, maxlen=max_len)
Xtest = pad_sequences(sequences[25000:50000], maxlen=max_len)
Ytrain = np.zeros((50000,), dtype=np.int8)
Ytest = np.zeros((25000,), dtype=np.int8)
Ytrain[25000:50000] = np.ones((25000,), dtype=np.int8)
Ytest[12500:25000] = np.ones((12500,), dtype=np.int8)


indice1 = np.arange(50000)
np.random.shuffle(indice1)
Xtrain = Xtrain[indice1]
Ytrain = Ytrain[indice1]

indice2 = np.arange(25000)
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
embedding = Embedding(num_words, 1000, input_length=max_len, embeddings_initializer='normal')(input)
x = AveragePooling1D(pool_size=3, strides=1)(embedding)
x = GlobalMaxPooling1D()(x)
# x=GlobalAveragePooling1D()(x)

# y=AveragePooling1D(pool_size=2,strides=1)(embedding)
# y=GlobalMaxPooling1D()(y)
# y=GlobalAveragePooling1D()(y)

# p=GlobalAveragePooling1D()(embedding)
# p=GlobalMaxPooling1D()(embedding)

# z=keras.layers.concatenate([y,p])
# z=keras.layers.maximum([y,p])
# x=Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, batch_size=32, epochs=50, validation_data=(Xtest, Ytest))
