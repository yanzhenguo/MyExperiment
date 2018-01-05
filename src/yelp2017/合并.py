import codecs
import json
import struct
import pickle
import numpy as np
import nltk

doc_vector = np.zeros((200000,4800),dtype=np.float32)


len_doc = np.load('D:\\docker\\Yelp2017\\SecondMethod\\lenDoc.npy')
# sumlen = 0
# for i in range(50000):
#     sumlen+=len_doc[i]
# print(sumlen)  #409291
# sumlen = 0
# for i in range(50000):
#     sumlen+=len_doc[i+50000]
# print(sumlen)  #411807  821098
# sumlen = 0
# for i in range(50000):
#     sumlen+=len_doc[i+100000]
# print(sumlen) #412334  1233432
# sumlen = 0
# for i in range(50000):
#     sumlen+=len_doc[i+150000]
# print(sumlen) #412259  1645691

sen_vector = np.load('D:\\docker\\Yelp2017\\SecondMethod\\SentencesVector1.npy')
j = 0
for i in range(50000):
    t_len = len_doc[i]
    temp_vector = np.zeros((4800),dtype=np.float32)
    count = 0
    for k in range(t_len):
        temp_vector+=sen_vector[j,:]
        j+=1
        count+=1
    temp_vector = temp_vector / count
    doc_vector[i,:] = temp_vector
    if i%1000==0:
        print(i)

j=0
sen_vector = np.load('D:\\docker\\Yelp2017\\SecondMethod\\SentencesVector2.npy')
for i in range(50000):
    t_len = len_doc[i+50000]
    temp_vector = np.zeros((4800),dtype=np.float32)
    count = 0
    for k in range(t_len):
        temp_vector+=sen_vector[j,:]
        j+=1
        count+=1
    temp_vector = temp_vector / count
    doc_vector[i+50000,:] = temp_vector
    if i%1000==0:
        print(i)

j=0
sen_vector = np.load('D:\\docker\\Yelp2017\\SecondMethod\\SentencesVector3.npy')
for i in range(50000):
    t_len = len_doc[i+100000]
    temp_vector = np.zeros((4800),dtype=np.float32)
    count = 0
    for k in range(t_len):
        temp_vector+=sen_vector[j,:]
        j+=1
        count+=1
    temp_vector = temp_vector / count
    doc_vector[i+100000,:] = temp_vector
    if i%1000==0:
        print(i)

j=0
sen_vector = np.load('D:\\docker\\Yelp2017\\SecondMethod\\SentencesVector4.npy')
for i in range(50000):
    t_len = len_doc[i+150000]
    temp_vector = np.zeros((4800),dtype=np.float32)
    count = 0
    for k in range(t_len):
        temp_vector+=sen_vector[j,:]
        j+=1
        count+=1
    temp_vector = temp_vector / count
    doc_vector[i+150000,:] = temp_vector
    if i%1000==0:
        print(i)

np.save('D:\\docker\\Yelp2017\\SecondMethod\\DocVector.npy',doc_vector)

