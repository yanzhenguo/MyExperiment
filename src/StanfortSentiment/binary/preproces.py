import codecs
import numpy as np
import pickle


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
# np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Y.npy',Y)
Y=np.load('D:\\docker\\StanfordSentiment\\FourthMethod\\Y.npy')        

leny = Y.shape[0]
for i in range(leny):
    if Y[i]<=0.4:
        Y[i] = 0
    elif Y[i]<=0.6:
        Y[i]=2
    else:
        Y[i]=1
f.close()



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
print(index)
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
print(index)
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
print(index)

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
#print(len(newY2))
#print(newtexts2[4000])
#print(newY2[4000])
out = codecs.open("D:\\docker\\StanfordSentiment\\FourthMethod\\texts.pkl",'wb')
pickle.dump(newtexts2,out,1)
out.close()
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Ytrain.npy',np.asarray(newY2)[0:6920])
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Ytest.npy',np.asarray(newY2)[6920:8741])
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Yval.npy',np.asarray(newY2)[8741:9613])
f.close()

# 6920 1821 872