import codecs
import numpy as np
import pickle

data_dir='/home/yan/PycharmProjects/MyExperiment/data/StanfordSentiment/'
temp_dir='../../../temp/StanfortSentiment/fine/get_data/'

#读取所有的句子到texts，读取其对应的情感得分到Y。
with open(data_dir+'datasetSentences.txt','r',encoding='utf-8') as f:
    texts = []
    text_dic={}
    count=0
    for line in f:
        if count==0:
            count=1
            continue
        sentence=line.split('\t')[1][:-1]
        texts.append(sentence)
        text_dic[sentence]=0


labledic = {}
count=0
flable = codecs.open(data_dir+'sentiment_labels.txt')
for line in flable:
    if count==0:
        count=1
        continue
    words = line.split('|')
    labledic[words[0]] = float(words[1][:-1])
flable.close()

Y = np.zeros((len(texts)),dtype=np.float32)
with open(data_dir+'dictionary.txt','r',encoding='utf-8') as f:
    for line in f:
        phrases = line.split('|')
        if phrases[0] in text_dic:
            text_dic[phrases[0]]=labledic[phrases[1][:-1]]
    for i in range(len(texts)):
        Y[i]=text_dic[texts[i]]
        # for index,line2 in enumerate(texts):
        #     if line2==phrases[0]:
        #         Y[index]=labledic[phrases[1][:-1]]

for i in range(Y.shape[0]):
    if Y[i]<=0.2:
        Y[i] = 0
    elif Y[i]<=0.4:
        Y[i]=1
    elif Y[i]<=0.6:
        Y[i]=2
    elif Y[i]<=0.8:
        Y[i]=3
    else:
        Y[i]=4


#将texts中的文本按照训练集、测试集、验证集的顺序排列
newtexts = []
newY = np.zeros((len(texts)),dtype=np.float32)
f=codecs.open(data_dir+'datasetSplit.txt')
count=0
index = 0
for line in f:
    if count==0:
        count=1
        continue        
    words = line[:-1].split(",")
    if words[1]=='1':
        newtexts.append(texts[int(words[0])-1])
        newY[index] = Y[int(words[0])-1]
        index+=1
count=0
f.seek(0)
for line in f:
    if count==0:
        count=1
        continue 
    words = line[:-1].split(",")
    if words[1]=='2':
        newtexts.append(texts[int(words[0])-1])
        newY[index] = Y[int(words[0])-1]
        index+=1
count=0
f.seek(0)
for line in f:
    if count==0:
        count=1
        continue  
    words = line[:-1].split(",")
    if words[1]=='3':
        newtexts.append(texts[int(words[0])-1])
        newY[index] = Y[int(words[0])-1]
        index+=1

#保存结果
# train:8544 test:2210 val:1090
out = codecs.open(temp_dir+'texts.pkl','wb')
pickle.dump(newtexts,out,1)
out.close()
np.save(temp_dir+'Ytrain.npy',newY[0:8544])
np.save(temp_dir+'Ytest.npy',newY[8544:10754])
np.save(temp_dir+'Yval.npy',newY[10754:11855])
f.close()

print(newY[:200])

