#prepare data
import codecs
import pickle
import numpy as np
from nltk.tokenize import WordPunctTokenizer
def extrac_data():
    data_dir='../../data/elec/'
    temp_dir='../../temp/elec/util/'
    texts=[]
    num_train = 25000
    num_test = 25000

    f=codecs.open(data_dir+'elec-25k-train.txt.tok','r','utf-8')
    for line in f:
        # line = line.replace('\\\\', '')
        # words = WordPunctTokenizer().tokenize(line.lower())
        texts.append(line.lower())
    f.close()

    f = codecs.open(data_dir+'elec-test.txt.tok','r','utf-8')
    for line in f:
        # line = line.replace('\\\\','')
        # words = WordPunctTokenizer().tokenize(line.lower())
        texts.append(line.lower())
    f.close()

    Ytrain=np.zeros((num_train,), dtype=np.int8)
    f = codecs.open(data_dir+'elec-25k-train.cat','r')
    index=0
    for line in f:
        Ytrain[index]=int(line[:-1])
        index+=1
    f.close()

    Ytest = np.zeros((num_test,), dtype=np.int8)
    f = codecs.open(data_dir+'elec-test.cat','r')
    index = 0
    for line in f:
        Ytest[index]=int(line[:-1])
        index+=1
    f.close()

    newText=[]
    for i in range(num_train):
        if Ytrain[i]==1:
            newText.append(texts[i])
    for i in range(num_train):
        if Ytrain[i]==2:
            newText.append(texts[i])
    for i in range(num_train,num_train+num_test):
        if Ytest[i-num_train]==1:
            newText.append(texts[i])
    for i in range(num_train,num_train+num_test):
        if Ytest[i-num_train]==2:
            newText.append(texts[i])

    pickle.dump(newText, open(temp_dir+'texts.pkl','wb'), 1)


if __name__=='__main__':
    extrac_data()


