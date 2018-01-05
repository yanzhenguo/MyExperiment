import codecs
import json
import struct
import pickle
import numpy as np
from nltk.tokenize import WordPunctTokenizer

texts = []
label = []
f = codecs.open('D:\我的下载\yelp_dataset_challenge_round9\\yelp_dataset_challenge_round9~\\yelp_academic_dataset_review.json','r','utf-8')

i=0
for line in f:
    j = json.loads(line)
    texts.append(j['text'])
    label.append(int(j['stars'])-1)
    i+=1
    if i%10000==0:
        print(i)
f.close()
#保存所有的样本到texts.pkl
f = codecs.open('D:\\docker\\Yelp2017\\FirstMethod\\texts.pkl','wb')
pickle.dump(texts,f,1)
f.close()

#保存所有样本的标签到label.npy
np.save("D:\\docker\\Yelp2017\\FirstMethod\\label.npy",np.asarray(label,dtype=np.int))
#print(len(label)) 4153150