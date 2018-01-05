import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D
num_words = 30000
max_len = 800
num_train = 11314
num_test = 7532
f = codecs.open('../temp/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[:num_train])
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
# sequences=[]
# for i in range(num_train+num_test):
#     t=[]
#     tokens=texts[i].lower().split(' ')
#     for j in range(len(tokens)):
#         index=word_index.get(tokens[j],0)
#         if index<num_words:
#             t.append(index)
#         else:
#             t.append(0)
#     sequences.append(t)

print('Found %s unique tokens.' % len(word_index))
Ytrain = np.load('../temp/Ytrain.npy')
Ytest = np.load('../temp/Ytest.npy')
Ytrain = to_categorical(Ytrain)
Ytest = to_categorical(Ytest)

data1 = pad_sequences(sequences[:num_train], maxlen=max_len)
data2 = pad_sequences(sequences[num_train:], maxlen=max_len)

Xtrain = np.zeros((num_train, (max_len-2)*3), dtype=np.int)
Xtest = np.zeros((num_test, (max_len-2)*3), dtype=np.int)
for i in range(num_train):
    for j in range(max_len-2):
        Xtrain[i, j*3] = data1[i, j]
        Xtrain[i, j*3+1] = data1[i][j+1]+num_words
        Xtrain[i, j*3+2] = data1[i][j+2]+num_words*2
for i in range(num_test):
    for j in range(max_len-2):
        Xtest[i, j*3] = data2[i, j]
        Xtest[i, j*3+1] = data2[i][j+1]+num_words
        Xtest[i, j*3+2] = data2[i][j+2]+num_words*2


indice1 = np.arange(num_train)
np.random.shuffle(indice1)
Xtrain = Xtrain[indice1]
Ytrain = Ytrain[indice1]

indice2 = np.arange(num_test)
np.random.shuffle(indice2)
Xtest = Xtest[indice2]
Ytest = Ytest[indice2]

print('begin to build model ...')
main_input = Input(shape=((max_len - 2)*3, ))
embedding1 = Embedding(num_words*3, 800, embeddings_initializer=keras.initializers.Orthogonal())(main_input)
x = AveragePooling1D(pool_size=3)(embedding1)
x = GlobalMaxPooling1D()(x)
output = Dense(20, activation='softmax')(x)


model = Model(inputs=main_input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit([Xtrain], Ytrain, batch_size=32, epochs=50, validation_data=([Xtest], Ytest))
