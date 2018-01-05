import numpy as np
import pickle
import codecs
from keras.datasets import reuters
word_index = reuters.get_word_index(path="reuters_word_index.json")

r=np.load('/home/yan/.keras/datasets/reuters.npz')
print(r.keys())
x = r['x']
y = r['y']

index_word = {}
for word,index in word_index.items():
    index_word[index] = word

texts = []
for i in range(len(x)):
    t = []
    for j in range(len(x[i])):
        t.append(index_word[x[i][j]])
    texts.append(' '.join(t))

f = codecs.open('../temp/text.pkl','wb')
pickle.dump(texts,f,1)
f.close()

np.save('../temp/Y.npy',y)