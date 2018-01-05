import codecs
import numpy as np
import pickle
#加载所有短语到texts,加载短语得分到字典labledic
print('process 1 ...')
f=codecs.open('D:\\docker\\StanfordSentiment\\data\\dictionary.txt','r','utf-8')
Y = []
texts = []
count=0
for line in f:
    if count==0:
        count=1
        continue
    words=line.strip('\n').split('|')
    texts.append(words[0])
    Y.append(float(words[1]))
f.close()
print(len(texts))
labledic = {}
count=0
flable = codecs.open('D:\\docker\\StanfordSentiment\\data\\sentiment_labels.txt')
for line in flable:
    if count==0:
        count=1
        continue
    words = line.split('|')
    labledic[words[0]] = float(words[1][:-1])
flable.close()

for i in range(len(Y)):
    Y[i] = labledic[str(int(Y[i]))]

#找出所有训练集中的句子,保存在train_sentence列表中
print('process 2 ...')
fsplit = codecs.open('D:\\docker\\StanfordSentiment\\data\\datasetSplit.txt','r','utf-8')
splitDic={}
count=0
for line in fsplit:
    if count==0:
        count=1
        continue
    words=line.strip('\n').split(',')
    splitDic[words[0]] = words[1]
fsplit.close()

f=codecs.open('D:\\docker\\StanfordSentiment\\data\\datasetSentences.txt','r','utf-8')
train_sentence = []
count=0
for line in f:
    if count==0:
        count=1
        continue
    tokens = line.strip('\n').split('\t')
    if splitDic[tokens[0]] == '1':
        train_sentence.append(tokens[1])
f.close()

#从texts中去除验证集和测试集的句子
print('process 3 ...')
#print(Y[0:200])
new_texts=[]
new_Y=[]
for index,line in enumerate(texts):
    if index%20000==0:
        print(index)
    flag = False
    for line2 in train_sentence:
        if line2.find(line)>=0:
            flag=True
    if flag:
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
print(new_Y[0:200])
#去除中性评价
print('process 5 ...')
new_texts2=[]
new_Y2=[]
for index,line in enumerate(new_texts):
    if index%10000==0:
        print(index)
    if new_Y[index]!=2:
        new_texts2.append(line)
        new_Y2.append(new_Y[index])
print(len(new_texts2))
#保存结果
print('save')
f=codecs.open('D:\\docker\\StanfordSentiment\\FourthMethod\\train_sentence.pkl','wb')
pickle.dump(new_texts2,f,1)
f.close()
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Ytrain.npy',np.asarray(new_Y2,dtype=np.int))


