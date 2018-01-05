import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D

num_words = 30000
max_len = 600
f = codecs.open('../temp/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.filters = ''
tokenizer.fit_on_texts(texts[0:25000])
# sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
sequences = []
for i in range(50000):
    t = []
    tokens = texts[i].lower().split(' ')
    for j in range(len(tokens)):
        index = word_index.get(tokens[j], 0)
        if index < num_words:
            t.append(index)
        else:
            t.append(0)
    sequences.append(t)

data1 = pad_sequences(sequences[0:25000], maxlen=max_len)
data2 = pad_sequences(sequences[25000:50000], maxlen=max_len)
Ytrain = np.zeros((25000,), dtype=np.int8)
Ytest = np.zeros((25000,), dtype=np.int8)
Ytrain[12500:25000] = np.ones((12500,), dtype=np.int8)
Ytest[12500:25000] = np.ones((12500,), dtype=np.int8)

Xtrain = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
Xtest = np.zeros((25000, (max_len - 2) * 3), dtype=np.int)
for i in range(25000):
    for j in range(max_len - 2):
        Xtrain[i, j * 3] = data1[i, j]
        Xtrain[i, j * 3 + 1] = data1[i][j + 1] + num_words
        Xtrain[i, j * 3 + 2] = data1[i][j + 2] + num_words * 2
        Xtest[i, j * 3] = data2[i, j]
        Xtest[i, j * 3 + 1] = data2[i][j + 1] + num_words
        Xtest[i, j * 3 + 2] = data2[i][j + 2] + num_words * 2
# np.save('../temp/Xtrain_seqcnn.npy',Xtrain)
# np.save('../temp/Xtest_seqcnn.npy',Xtest)

indice1 = np.arange(25000)
np.random.shuffle(indice1)
Xtrain = Xtrain[indice1]
Ytrain = Ytrain[indice1]

indice2 = np.arange(25000)
np.random.shuffle(indice2)
Xtest = Xtest[indice2]
Ytest = Ytest[indice2]

print('begin to build model ...')
main_input = Input(shape=((max_len - 2) * 3,))
embedding1 = Embedding(num_words * 3, 800, embeddings_initializer=keras.initializers.Orthogonal())(main_input)
x = AveragePooling1D(pool_size=3)(embedding1)
x = GlobalMaxPooling1D()(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit([Xtrain], Ytrain, batch_size=32, epochs=50, validation_data=([Xtest], Ytest))

# model.save('../temp/seqCNN_3_model.h5')
