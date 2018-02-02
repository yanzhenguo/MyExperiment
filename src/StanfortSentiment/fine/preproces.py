import codecs
import numpy as np
import pickle

#读取所有的句子到texts，读取其对应的情感得分到Y。
f=codecs.open('D:\\docker\\StanfordSentiment\\data\\datasetSentences.txt','r','utf-8')
texts = []
count=0
for line in f:
    if count==0:
        count=1
        continue
    texts.append(line.split('\t')[1][:-1])
f.close()

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

# Y = np.zeros((len(texts)),dtype=np.float32)
# f=codecs.open('D:\\docker\\StanfordSentiment\\data\\dictionary.txt','r','utf-8')
# for line in f:
#     phrases = line.split('|')
#     for index,line2 in enumerate(texts):
#         if line2==phrases[0]:
#             Y[index]=labledic[phrases[1][:-1]]
# np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Y.npy',Y)
Y=np.load('D:\\docker\\StanfordSentiment\\FirstMethod\\Y.npy')
leny = Y.shape[0]

for i in range(leny):
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
f.close()


#将texts中的文本按照训练集、测试集、验证集的顺序排列
newtexts = []
newY = np.zeros((len(texts)),dtype=np.float32)
f=codecs.open('D:\\docker\\StanfordSentiment\\data\\datasetSplit.txt')
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
out = codecs.open("D:\\docker\\StanfordSentiment\\FirstMethod\\texts.pkl",'wb')
pickle.dump(newtexts,out,1)
out.close()
np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Ytrain.npy',newY[0:8544])
np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Ytest.npy',newY[8544:10754])
np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Yval.npy',newY[10754:11855])
f.close()

