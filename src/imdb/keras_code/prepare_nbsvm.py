import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

num_words=100000
f = codecs.open('../FirstMethod/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()

tokenizer = Tokenizer(num_words=num_words)
tokenizer.filters=''
tokenizer.fit_on_texts(texts[0:25000])
word_count=tokenizer.word_counts
sequences = tokenizer.texts_to_sequences(texts)

# max_id=0
# for i in range(50000):
#     if sequences[i][-1]>max_id:
#         max_id=sequences[i][-1]
# print(max_id)

# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))


#count 1-gram
Ytrain = np.zeros((25000,),dtype=np.float32)
Ytest = np.zeros((25000,),dtype=np.float32)
Ytrain[12500:25000]=np.ones((12500,),dtype=np.float32)
Ytest[12500:25000]=np.ones((12500,),dtype=np.float32)

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
new_gram={}
index=num_words
for i in range(25000):
    for j in range(len(sequences[i])-1):
        if tuple(sequences[i][j:j+2]) not in new_gram:
            new_gram[tuple(sequences[i][j:j+2])] = index
            index+=1

sequences2=[]
for i in range(50000):
    t=[]
    for j in range(len(sequences[i])-1):
        if tuple(sequences[i][j:j+2]) in new_gram:
            t.append(new_gram[tuple(sequences[i][j:j+2])])
    sequences2.append(t)
#count 3-gram
new_gram = {}
for i in range(25000):
    for j in range(len(sequences[i])-2):
        if tuple(sequences[i][j:j+3]) not in new_gram:
            new_gram[tuple(sequences[i][j:j+3])] = index
            index+=1

sequences3=[]
for i in range(50000):
    t=[]
    for j in range(len(sequences[i])-2):
        if tuple(sequences[i][j:j+3]) in new_gram:
            t.append(new_gram[tuple(sequences[i][j:j+3])])
    sequences3.append(t)

for i in range(50000):
    sequences[i]= list(set(sequences[i]))
    sequences[i].sort()
    sequences2[i]= list(set(sequences2[i]))
    sequences2[i].sort()
    sequences3[i]= list(set(sequences3[i]))
    sequences3[i].sort()
# indice2 = np.arange(25000)
# np.random.shuffle(indice2)
# Xtest=Xtest[indice2]
# Ytest=Ytest[indice2]

#write to file
print('begin to write')
outXtrain=codecs.open('Xtrain.txt','w','utf-8')
text=[]
for i in range(25000):
    line=''
    if Ytrain[i]==0:
        line+='-1 '
    else:
        line+='1 '
    #1-gram
    len1=len(sequences[i])
    len2 = len(sequences2[i])
    len3= len(sequences3[i])
    len123=len1+len2
    for j in range(len1):
        line += str(sequences[i][j]) + ':'+str(1/len123**(0.5)) +' '
    #2-gram
    for j in range(len2):
        line += str(sequences2[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    # #3-gram
    for j in range(len3):
        line += str(sequences3[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    line=line[:-1]
    line+='\n'
    text.append(line)
    if i%1000==0:
        print(i)
for i in range(25000):
    outXtrain.write(text[i])
outXtrain.close()

outXtest=codecs.open('Xtest.txt','w','utf-8')
text=[]
for i in range(25000,50000):
    line=''
    if Ytest[i-25000]==0:
        line+='-1 '
    else:
        line+='1 '
    # 1-gram
    len1 = len(sequences[i])
    len2 = len(sequences2[i])
    len3 = len(sequences3[i])
    len123 = len1 + len2+len3
    for j in range(len1):
        line += str(sequences[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    # 2-gram
    for j in range(len2):
        line += str(sequences2[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    # 3-gram
    for j in range(len3):
        line += str(sequences3[i][j]) + ':' + str(1 / len123 ** (0.5)) + ' '
    line = line[:-1]
    line += '\n'
    text.append(line)
    if i%1000==0:
        print(i)
for i in range(25000):
    outXtest.write(text[i])
outXtest.close()
