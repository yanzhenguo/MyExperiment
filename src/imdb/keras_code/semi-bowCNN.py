import codecs
import pickle
import keras
import numpy as np
from keras.layers import Dense, GlobalMaxPooling1D, Input, Embedding, \
    AveragePooling1D, GlobalAveragePooling1D, Activation
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

num_words = 30000
max_len = 500
num_train = 25000
num_test = 25000
f = codecs.open('../temp/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.filters = ''
tokenizer.fit_on_texts(texts[:num_train])

num_pos = 0
num_neg = 0

sequences = tokenizer.texts_to_sequences(texts)
train_sequences = []
for i in range(25000):
    for j in range(len(sequences[i]) - 4):
        train_sequences.append(sequences[i][j:j + 5])
        if i < 12500:
            num_pos += 1
        else:
            num_neg += 1

train_sequences = np.asarray(train_sequences, dtype=np.int)
word_index = tokenizer.word_index

# sequences=[]
# for i in range(50000):
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

Ytrain = np.zeros((len(train_sequences),), dtype=np.int8)
Ytrain[num_pos:] = np.ones((num_neg,), dtype=np.int8)

indice1 = np.arange(len(train_sequences))
np.random.shuffle(indice1)
Xtrain = train_sequences[indice1]
Ytrain = Ytrain[indice1]

main_input = Input(shape=(5,))
# model.add(Embedding(35000,50,input_length=500))
init_method = keras.initializers.normal
# embedding1 = Embedding(num_words, 500, embeddings_initializer=init_method)(main_input)
x = Embedding(num_words, 200, embeddings_initializer=init_method)(main_input)
# x = AveragePooling1D(pool_size=3, strides=1,padding='valid')(x)
# x = Activation('relu')(x)
# x = GlobalMaxPooling1D()(x)
x = GlobalAveragePooling1D()(x)
x = Activation('relu')(x)

# embedding2=Embedding(num_words,50, embeddings_initializer=init_method)(main_input)
# y=AveragePooling1D(pool_size=2,strides=1)(embedding2)
# y=GlobalMaxPooling1D()(y)

# embedding3=Embedding(num_words,50,input_length=max_len,embeddings_initializer='normal')(input)
# p=GlobalAveragePooling1D()(embedding3)

# z=keras.layers.concatenate([x,y])
# z=keras.layers.concatenate([x,y,p])
# x=Dropout(0.2)(x)
output = Dense(1, activation='sigmoid', trainable=True, use_bias=False)(x)

model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, batch_size=512, epochs=20)

# predict = model.predict(Xtest)
# out = open('../temp/prediction.txt', 'w')
# for i in range(25000):
#     if (Ytest[i] == 0 and predict[i, 0] > 0.5) or (Ytest[i] == 1 and predict[i, 0] < 0.5):
#         out.write(str(int(Ytest[i])) + ' ' + str(predict[i, 0]) + '\n')
# out.close()
model.summary()
model.save('../temp/semi-bowCNN.model.h5')
