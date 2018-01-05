import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

num_words=20000
num_train=11314
num_test=7532
f = codecs.open('train.pkl', 'rb')
train_text = pickle.load(f)
f.close()
f = codecs.open('test.pkl', 'rb')
test_text = pickle.load(f)
f.close()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.filters=''
tokenizer.fit_on_texts(train_text)
word_index = tokenizer.word_index
word_count=tokenizer.word_counts
count_matrix=tokenizer.texts_to_matrix(train_text+test_text,'count')
count_matrix=np.log(count_matrix+1)
# sequences = tokenizer.texts_to_sequences(train_text+test_text)

# frequency_dic={}
# for word,id in word_index.items():
#     if id<num_words:
#         frequency_dic[id]=word_count[word]
# max_id=0
# for i in range(50000):
#     if sequences[i][-1]>max_id:
#         max_id=sequences[i][-1]
# print(max_id)

# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))


#count 1-gram
Ytrain = np.load('Ytrain.npy')
Ytest = np.load('Ytest.npy')
# indice1 = np.arange(25000)
# np.random.shuffle(indice1)
# Xtrain=Xtrain[indice1]
# Ytrain=Ytrain[indice1]
#


# np.save('Xtrain.npy',Xtrain)
# np.save('Xtest.npy',Xtest)
# np.save('Ytrain.npy',Ytrain)
# np.save('Ytest.npy',Ytest)



#count 2-gram
# new_gram={}
# index=num_words
# for i in range(num_train):
#     for j in range(len(sequences[i])-1):
#         if tuple(sequences[i][j:j+2]) not in new_gram:
#             new_gram[tuple(sequences[i][j:j+2])] = index
#             frequency_dic[index]=1
#             index+=1
#         else:
#             frequency_dic[new_gram[tuple(sequences[i][j:j+2])]]+=1
#
# sequences2=[]
# for i in range(num_train+num_test):
#     t=[]
#     for j in range(len(sequences[i])-1):
#         if tuple(sequences[i][j:j+2]) in new_gram:
#             t.append(new_gram[tuple(sequences[i][j:j+2])])
#     sequences2.append(t)
#count 3-gram
# new_gram = {}
# for i in range(num_train):
#     for j in range(len(sequences[i])-2):
#         if tuple(sequences[i][j:j+3]) not in new_gram:
#             new_gram[tuple(sequences[i][j:j+3])] = index
#             frequency_dic[index]=1
#             index+=1
#         else:
#             frequency_dic[new_gram[tuple(sequences[i][j:j+3])]] +=1
#
# sequences3=[]
# for i in range(num_train+num_test):
#     t=[]
#     for j in range(len(sequences[i])-2):
#         if tuple(sequences[i][j:j+3]) in new_gram:
#             t.append(new_gram[tuple(sequences[i][j:j+3])])
#     sequences3.append(t)

# for i in range(num_train+num_test):
#     sequences[i]= list(set(sequences[i]))
#     sequences[i].sort()
    # sequences2[i]= list(set(sequences2[i]))
    # sequences2[i].sort()
    # sequences3[i]= list(set(sequences3[i]))
    # sequences3[i].sort()
# indice2 = np.arange(25000)
# np.random.shuffle(indice2)
# Xtest=Xtest[indice2]
# Ytest=Ytest[indice2]

#write to file
print('begin to write')
outXtrain=codecs.open('Xtrain.txt','w','utf-8')
text=[]
for i in range(num_train):
    line=''
    line+=str(int(Ytrain[i])+1)+' '
    #1-gram
    len1=np.sum(count_matrix[i])
    for j in range(num_words):
        if count_matrix[i,j]!=0:
            line += str(j+1) + ':'+str(count_matrix[i, j]/len1**(0.5)) +' '
    #2-gram
    # for j in range(len2):
    #     line += str(sequences2[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    # #3-gram
    # for j in range(len3):
    #     line += str(sequences3[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    line=line[:-1]
    line+='\n'
    text.append(line)
    if i%1000==0:
        print(i)
for i in range(num_train):
    outXtrain.write(text[i])
outXtrain.close()

outXtest=codecs.open('Xtest.txt','w','utf-8')
text=[]
for i in range(num_train,num_train+num_test):
    line=''
    line+=str(int(Ytest[i-num_train])+1)+' '
    # 1-gram
    len1 = np.sum(count_matrix[i])
    for j in range(num_words):
        if count_matrix[i, j] != 0:
            line += str(j+1) + ':' + str(count_matrix[i, j] / len1 ** (0.5)) + ' '
    # 2-gram
    # for j in range(len2):
    #     line += str(sequences2[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    # 3-gram
    # for j in range(len3):
    #     line += str(sequences3[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    line = line[:-1]
    line += '\n'
    text.append(line)
    if i%1000==0:
        print(i)
for i in range(num_test):
    outXtest.write(text[i])
outXtest.close()
