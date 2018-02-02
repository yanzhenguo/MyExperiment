# -*- coding: utf-8 -*-

# 提取短语级别的训练样本

import codecs
import numpy as np
import pickle

data_dir='../../../data/StanfordSentiment/'
temp_dir='../../../temp/StanfortSentiment/binary/processTrain/'

#加载所有短语到texts,加载短语得分到字典labledic
print('process 1 ...')
Y = []
texts = []
with open(data_dir+'dictionary.txt') as f:
    next(f)
    for line in f:
        words=line.strip('\n').split('|')
        texts.append(words[0])
        Y.append(float(words[1]))
    print(len(texts))

labledic = {}
with open(data_dir+'sentiment_labels.txt') as f:
    next(f)
    for line in f:
        words = line.split('|')
        labledic[words[0]] = float(words[1][:-1])


for i in range(len(Y)):
    Y[i] = labledic[str(int(Y[i]))]

#找出所有训练集中的句子,保存在train_sentence列表中
print('process 2 ...')
splitDic={}
with open(data_dir+'datasetSplit.txt') as f:
    next(f)
    for line in f:
        words=line.strip('\n').split(',')
        splitDic[words[0]] = words[1]


train_sentence = ''
with open(data_dir+'datasetSentences.txt') as f:
    next(f)
    for line in f:
        tokens = line.strip('\n').split('\t')
        if splitDic[tokens[0]] == '1':
            train_sentence+=tokens[1]

#从texts中去除验证集和测试集的句子
print('process 3 ...')
new_texts=[]
new_Y=[]
for index,line in enumerate(texts):
    if index%20000==0:
        print(index)
    # flag = False
    # for line2 in train_sentence:
    #     if line2.find(line)>=0:
    #         flag=True
    # if flag:
    #     new_texts.append(line)
    #     new_Y.append(Y[index])
    if train_sentence.find(line)>0:
        new_texts.append(line)
        new_Y.append(Y[index])
print(len(new_texts))
#print(new_Y[0:200])

#计算每个训练样本的类别
print('process 4 ...')
for i in range(len(new_Y)):
    if new_Y[i]<=0.4:
        new_Y[i] = 0    
    elif new_Y[i]<=0.6:
        new_Y[i]=2
    else:
        new_Y[i]=1
# print(new_Y[0:200])

#去除中性评价
print('process 5 ...')
new_texts2=[]
new_Y2=[]
for index,line in enumerate(new_texts):
    if new_Y[index]!=2:
        new_texts2.append(line)
        new_Y2.append(new_Y[index])
print(len(new_texts2))

#保存结果
print('save...')
with open(temp_dir+'train_sentence.pkl','wb') as f:
    pickle.dump(new_texts2,f,1)

np.save(temp_dir+'Ytrain.npy',np.asarray(new_Y2,dtype=np.int))


