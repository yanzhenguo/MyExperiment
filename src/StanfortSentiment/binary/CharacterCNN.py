import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

f = codecs.open('D:\\docker\\StanfordSentiment\\FourthMethod\\texts.pkl','rb')
texts = pickle.load(f)
f.close()

chardic={}
index=1
for line in texts:
    line=line.lower()
    for c in line:
        if c not in chardic:
            chardic[c]=index
            index+=1
sequences = []
for line in texts:
    line=line.lower()
    temp=[]
    for c in line:
        temp.append(chardic[c])
    sequences.append(temp)

Xtrain = pad_sequences(sequences[0:7077], maxlen=300)
newXtrain = np.zeros((7077,300,74),dtype=np.int)
for i in Xtrain:
    ii=np.asarray(i,dtype=np.int)
    ii=to_categorical(ii,74)
    newXtrain[i,:,:]=ii
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Xtrain.npy',newXtrain)

Xtest = pad_sequences(sequences[7077:8826], maxlen=300)
newXtest = np.zeros((1749,300,74),dtype=np.int)
for i in Xtest:
    ii=np.asarray(i,dtype=np.int)
    ii=to_categorical(ii,74)
    newXtest[i,:,:]=ii
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Xtest.npy',newXtest)
