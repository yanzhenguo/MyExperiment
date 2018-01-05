import codecs
import json
import struct
import pickle
import numpy as np
import nltk
sent_tokenizer = nltk.data.load('tokenizers\\punkt\\english.pickle')

f = codecs.open('D:\\docker\\Yelp2017\\FirstMethod\\texts.pkl','rb')
texts = pickle.load(f)
f.close()

sentences = []  #保存所有的句子
len_ss = []     #保存每篇文章对应的句子数
len_texts=200000
for i in range(len_texts):
    ss = sent_tokenizer.tokenize(texts[i])
    sentences+=ss
    len_ss.append(len(ss))
    if i%1000==0:
        print(i)

f = codecs.open('D:\\docker\\Yelp2017\\SecondMethod\\sentences.pkl','wb')
pickle.dump(sentences,f,1)
f.close()

np.save('D:\\docker\\Yelp2017\\SecondMethod\\lenDoc.npy',np.asarray(len_ss,dtype=np.int))