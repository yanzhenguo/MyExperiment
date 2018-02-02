# -*- coding: utf-8 -*-

# 提取句子级别的样本和标签

import numpy as np
import pickle

data_dir='../../../data/StanfordSentiment/'
temp_dir='../../../temp/StanfortSentiment/binary/preprocess/'

with open(data_dir+'datasetSentences.txt',mode='r',encoding='utf-8') as f:
    texts = []
    count=0
    for line in f:
        if count==0:
            count=1
            continue
        texts.append(line.split('\t')[1][:-1])


labledic = {}
count=0
with open(data_dir+'sentiment_labels.txt') as f:
    for line in f:
        if count==0:
            count=1
            continue
        words = line.split('|')
        labledic[words[0]] = float(words[1][:-1])


Y = np.zeros((len(texts)),dtype=np.float32)
phrase_id=dict()
with open(data_dir+'dictionary.txt') as f:
    for line in f:
        phrases = line.split('|')
        phrase_id[phrases[0]]=phrases[1][:-1]

for doc_id,doc in enumerate(texts):
    Y[doc_id]=labledic[phrase_id[doc]]

print('has got Y')

for i in range(len(texts)):
    if Y[i]<=0.4:
        Y[i] = 0
    elif Y[i]<=0.6:
        Y[i]=2
    else:
        Y[i]=1


newtexts = []
newY = np.zeros((len(texts)),dtype=np.float32)
with open(data_dir+'datasetSplit.txt') as f:
    train_id=[]
    test_id=[]
    val_id=[]
    lines=f.readlines()[1:]
    for line in lines:
        words = line[:-1].split(",")
        if words[1] == '1':
            train_id.append(int(words[0])-1)
        elif words[1] == '2':
            test_id.append(int(words[0]) - 1)
        else:
            val_id.append(int(words[0])-1)
    for i,id in enumerate(train_id+test_id+val_id):
        newtexts.append(texts[id])
        newY[i]=Y[id]


# 去除中性句子
newtexts2=[]
newY2=[]
count=0
for i in range(8544):
    if newY[i]!=2:
        count+=1
        newY2.append(newY[i])
        newtexts2.append(newtexts[i])
print(count)
count=0
for i in range(2210):
    if newY[i+8544]!=2:
        count+=1
        newY2.append(newY[i+8544])
        newtexts2.append(newtexts[i+8544])
print(count)
count=0
for i in range(1101):
    if newY[i+10754]!=2:
        count+=1
        newY2.append(newY[i+10754])
        newtexts2.append(newtexts[i+10754])
print(count)

with open(temp_dir+"texts.pkl",'wb') as f:
    pickle.dump(newtexts2,f,1)

np.save(temp_dir+'Ytrain.npy',np.asarray(newY2)[0:6920])
np.save(temp_dir+'Ytest.npy',np.asarray(newY2)[6920:8741])
np.save(temp_dir+'Yval.npy',np.asarray(newY2)[8741:9613])


# 6920 1821 872