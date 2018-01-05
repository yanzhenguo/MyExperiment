import codecs
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

num_words=30000
max_len=1000
num_train=11314
num_test=7532
f = codecs.open('../temp/texts.pkl','rb')
texts=pickle.load(f)
f.close()


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[:num_train])
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
#print(np.sum(np.asarray([len(s.split(' ')) for s in texts])) / len(texts))
data1 = pad_sequences(sequences[:num_train], maxlen=max_len)
data2 = pad_sequences(sequences[num_train:], maxlen=max_len)
labels1 = np.load('../temp/Ytrain.npy')
labels2 = np.load('../temp/Ytest.npy')


embeddings_index = {}
wordX = np.load('/media/yan/winD/docker/word2vec/words/word2vec.npy')
f= codecs.open('/media/yan/winD/docker/word2vec/words/wordsInWord2vec.pkl','rb')
allwords = pickle.load(f)
f.close()
for i in range(3000000):
    embeddings_index[''.join(allwords[i])] = wordX[i,:]
# f=codecs.open('/media/yan/winD/docker/glove6B/glove.6B.100d.pkl','rb')
# embeddings_index=pickle.load(f)
# f.close()

embedding_matrix = np.zeros((num_words, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None and i<num_words:
        embedding_matrix[i] = embedding_vector

np.save('../temp/Xtrain_cnn.npy',data1)
np.save('../temp/Xtest_cnn.npy',data2)

np.save('../temp/embeddingMatrix.npy',embedding_matrix)