import codecs
import pickle
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D
num_words=20000
max_len=50
f = codecs.open('../FirstMethod/texts.pkl','rb')
texts = pickle.load(f)
f.close()


Ytrain=np.load('../FirstMethod/Ytrain.npy')
Ytest=np.load('../FirstMethod/Ytest.npy')
Ytrain=to_categorical(Ytrain)
Ytest=to_categorical(Ytest)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[0:8544])

#sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

sequences=[]
for i in range(10754):
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

data1 = pad_sequences(sequences[0:8544], maxlen=max_len)
data2 = pad_sequences(sequences[8544:10754], maxlen=max_len)

indice1 = np.arange(len(data1))
np.random.shuffle(indice1)
Xtrain=data1[indice1]
Ytrain=Ytrain[indice1]

indice2 = np.arange(len(data2))
np.random.shuffle(indice2)
Xtest=data2[indice2]
Ytest=Ytest[indice2]

# np.save('Xtrain',Xtrain)
# np.save('Xtest',Xtest)
# np.save('Ytrain',Ytrain)
# np.save('Ytest',Ytest)

# Xtrain=np.load('Xtrain.npy')
# Xtest=np.load('Xtest.npy')
# Ytrain=np.load('Ytrain.npy')
# Ytest=np.load('Ytest.npy')

input=Input(shape=(max_len,))
# embedding1=Embedding(num_words,50,input_length=max_len,embeddings_initializer='normal')(input)
# x=AveragePooling1D(pool_size=3,strides=1)(embedding1)
# x=GlobalMaxPooling1D()(x)
#x=GlobalAveragePooling1D()(x)
# embedding2=Embedding(num_words,500,input_length=max_len,embeddings_initializer='normal')(input)
# y=AveragePooling1D(pool_size=2,strides=1)(embedding2)
# y=GlobalMaxPooling1D()(y)

embedding3=Embedding(num_words,100,input_length=max_len,embeddings_initializer='normal')(input)
p=GlobalAveragePooling1D()(embedding3)
#p=GlobalMaxPooling1D()(embedding3)

#z=keras.layers.concatenate([x,y])
#z=keras.layers.maximum([x,y,p])
p=Dropout(0.5)(p)
output=Dense(5,activation='softmax')(p)

model=Model(inputs=input,outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(Xtrain, Ytrain,batch_size=32,epochs=50,validation_data=(Xtest, Ytest))