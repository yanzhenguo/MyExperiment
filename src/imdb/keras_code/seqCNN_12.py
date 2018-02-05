import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D
num_words=30000
max_len=800
f = codecs.open('../FirstMethod/texts.pkl','rb')
#f = codecs.open('texts.pkl','rb')
texts = pickle.load(f)
f.close()

# for i in range(50000):
#     texts[i]=texts[i].lower()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[0:25000])
#print(texts[0])
#sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
sequences=[]
for i in range(50000):
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

data1 = pad_sequences(sequences[0:25000], maxlen=max_len)
data2 = pad_sequences(sequences[25000:50000], maxlen=max_len)
Ytrain = np.zeros((25000,),dtype=np.float32)
Ytest = np.zeros((25000,),dtype=np.float32)
Ytrain[12500:25000]=np.ones((12500,),dtype=np.float32)
Ytest[12500:25000]=np.ones((12500,),dtype=np.float32)

Xtrain1=data1
Xtest1=data2


Xtrain2=np.zeros((25000,(max_len-1)*2),dtype=np.int)
Xtest2=np.zeros((25000,(max_len-1)*2),dtype=np.int)
for i in range(25000):
    for j in range(max_len-1):
        Xtrain2[i,j*2]=data1[i,j]
        Xtrain2[i,j*2+1] = data1[i][j+1]+num_words
for i in range(25000):
    for j in range(max_len-1):
        Xtest2[i,j*2]=data2[i,j]
        Xtest2[i,j*2+1] = data2[i][j+1]+num_words

indice1 = np.arange(25000)
np.random.shuffle(indice1)
Xtrain1=Xtrain1[indice1]
Xtrain2=Xtrain2[indice1]
Ytrain=Ytrain[indice1]

indice2 = np.arange(25000)
np.random.shuffle(indice2)
Xtest1=Xtest1[indice2]
Xtest2=Xtest2[indice2]
Ytest=Ytest[indice2]
print('begin to build model ...')
input1=Input(shape=(max_len,))
embedding1=Embedding(num_words,500,input_length=max_len,embeddings_initializer='normal')(input1)
x=GlobalAveragePooling1D()(embedding1)

input2=Input(shape=((max_len-1)*2,))
embedding2=Embedding(num_words*2,500,input_length=(max_len-1)*2,embeddings_initializer='normal')(input2)
y=AveragePooling1D(pool_size=2,strides=2)(embedding2)
y=GlobalMaxPooling1D()(y)
z=keras.layers.concatenate([x,y])
#model.add(Dropout(0.5))
output=Dense(1,activation='sigmoid')(z)

model=Model(inputs=[input1,input2],outputs=output)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit([Xtrain1,Xtrain2], Ytrain,batch_size=32,epochs=50,validation_data=([Xtest1,Xtest2], Ytest))

# model=Model(inputs=input2,outputs=output)
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit([Xtrain2], Ytrain,batch_size=32,epochs=50,validation_data=([Xtest2], Ytest))