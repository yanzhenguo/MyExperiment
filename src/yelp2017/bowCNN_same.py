import codecs
import pickle
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D,MaxPooling1D,Flatten
#in this script, different n-grams share the same word embedding
num_words=30000
max_len=200
f = codecs.open('../FirstMethod/texts.pkl','rb')
#f = codecs.open('texts.pkl','rb')
texts = pickle.load(f)
f.close()

# for i in range(50000):
#     texts[i]=texts[i].lower()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[0:160000])
#print(texts[0])
#sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

sequences=[]
for i in range(200000):
    t=[]
    tokens=texts[i].lower().split(' ')
    for j in range(len(tokens)):
        index=word_index.get(tokens[j],0)
        if index<num_words:
            t.append(index)
        else:
            t.append(0)
    sequences.append(t)

print('Found %s unique tokens.' % len(word_index))
sumlen=0
for i in range(160000):
    sumlen+=len(sequences[i])
print(sumlen/160000)
Xtrain = pad_sequences(sequences[0:160000], maxlen=max_len)
Xtest = pad_sequences(sequences[160000:200000], maxlen=max_len)
Ytrain = np.load('../FirstMethod/label.npy')[0:160000]
Ytest = np.load('../FirstMethod/label.npy')[160000:200000]
#print(Ytrain[0:50])
Ytrain=to_categorical(Ytrain)
Ytest=to_categorical(Ytest)
# indice1 = np.arange(25000)
# np.random.shuffle(indice1)
# Xtrain=data1[indice1]
# Ytrain=Ytrain[indice1]
#
# indice2 = np.arange(25000)
# np.random.shuffle(indice2)
# Xtest=data2[indice2]
# Ytest=Ytest[indice2]

# np.save('Xtrain',Xtrain)
# np.save('Xtest',Xtest)
# np.save('Ytrain',Ytrain)
# np.save('Ytest',Ytest)

# Xtrain=np.load('Xtrain.npy')
# Xtest=np.load('Xtest.npy')
# Ytrain=np.load('Ytrain.npy')
# Ytest=np.load('Ytest.npy')

input=Input(shape=(max_len,))
embedding=Embedding(num_words,1000,input_length=max_len,embeddings_initializer='normal')(input)
x=AveragePooling1D(pool_size=3,strides=1)(embedding)
x=GlobalMaxPooling1D()(x)
#x=GlobalAveragePooling1D()(x)

# y=AveragePooling1D(pool_size=2,strides=1)(embedding)
# y=GlobalMaxPooling1D()(y)
#y=GlobalAveragePooling1D()(y)

#p=GlobalAveragePooling1D()(embedding)
#p=GlobalMaxPooling1D()(embedding)

#z=keras.layers.concatenate([y,p])
#z=keras.layers.maximum([y,p])
#x=Dropout(0.2)(x)
output=Dense(5,activation='softmax')(x)

model=Model(inputs=input,outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(Xtrain, Ytrain,batch_size=64,epochs=50,validation_data=(Xtest, Ytest))