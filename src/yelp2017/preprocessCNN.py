import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import WordPunctTokenizer

basedir = 'D:\\docker\\Yelp2017\\FirstMethod\\'
#-----------------------------------------------------------------------------------
f = codecs.open(basedir+'texts.pkl','rb')
texts = pickle.load(f)
f.close()
texts = texts[0:200000]

#-----------------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Xall = pad_sequences(sequences, maxlen=1000)

#建立词向量矩阵
wordX = np.load('D:\\docker\\word2vec\\words\\word2vec.npy')
f= codecs.open('D:\\docker\\word2vec\\words\\wordsInWord2vec.pkl','rb')
allwords = pickle.load(f)
f.close()
embeddings_index = {}
for i in range(3000000):
    embeddings_index[''.join(allwords[i])] = wordX[i,:]
# for line in fWord:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
#保存结果
np.save(basedir+'Xall.npy',Xall)
np.save(basedir+'embeddingMatrix.npy',embedding_matrix)