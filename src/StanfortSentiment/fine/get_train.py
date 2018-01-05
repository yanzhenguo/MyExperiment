import codecs
import numpy as np
import pickle
#加载所有短语到texts,加载短语得分到字典labledic

data_dir='/home/yan/PycharmProjects/MyExperiment/data/StanfordSentiment/'
temp_dir='../../../temp/StanfortSentiment/fine/get_train/'

print('process 1 ...')
f=codecs.open(data_dir+'dictionary.txt','r','utf-8')
Y = []
texts = []
for line in f:
    words=line.strip('\n').split('|')
    texts.append(words[0])
    Y.append(float(words[1]))
f.close()
print(len(texts))

labledic = {}
flable = codecs.open(data_dir+'sentiment_labels.txt')
flable.readline()
for line in flable:
    words = line.split('|')
    labledic[words[0]] = float(words[1][:-1])
flable.close()

for i in range(len(Y)):
    Y[i] = labledic[str(int(Y[i]))]

#找出所有的训练集中的句子,保存在train_sentence列表中
print('process 2 ...')
fsplit = codecs.open(data_dir+'datasetSplit.txt','r','utf-8')
fsplit.readline()
splitDic={}
for line in fsplit:
    words=line.strip('\n').split(',')
    splitDic[words[0]] = words[1]
fsplit.close()

with open(data_dir+'datasetSentences.txt','r',encoding='utf-8') as f:
    f.readline()
    train_sentence = []
    for line in f:
        tokens = line.strip('\n').split('\t')
        if splitDic[tokens[0]] == '1':
            train_sentence.append(tokens[1])
train_text=' '.join(train_sentence)
print(len(train_sentence))

#从texts中去除验证集和测试集的句子
print('process 3 ...')
#print(Y[0:200])
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
    if train_text.find(line)>=0:
        new_texts.append(line)
        new_Y.append(Y[index])
print(len(new_texts))


#计算每个训练样本的类别
print('process 4 ...')
for i in range(len(new_Y)):
    if new_Y[i]<=0.2:
        new_Y[i] = 0
    elif new_Y[i]<=0.4:
        new_Y[i]=1
    elif new_Y[i]<=0.6:
        new_Y[i]=2
    elif new_Y[i]<=0.8:
        new_Y[i]=3
    else:
        new_Y[i]=4

#保存结果
f=codecs.open(temp_dir+'train_sentence.pkl','wb')
pickle.dump(new_texts,f,1)
f.close()
np.save(temp_dir+'Ytrain.npy',np.asarray(new_Y,dtype=np.int))


