import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

num_words = 30000
max_len = 500
f = codecs.open('../temp/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts[0:25000])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

words = ['' for i in range(29999)]
for word, id in word_index.items():
    if id < 30000:
        words[id-1] = word
f = codecs.open('../temp/words.pkl', 'wb')
pickle.dump(words, f, 1)
f.close()

data1 = pad_sequences(sequences[0:25000], maxlen=max_len)
data2 = pad_sequences(sequences[25000:50000], maxlen=max_len)
labels1 = np.zeros((25000,), dtype=np.float32)
labels2 = np.zeros((25000,), dtype=np.float32)
labels1[0:12500] = np.ones((12500,), dtype=np.float32)
labels2[0:12500] = np.ones((12500,), dtype=np.float32)

indice1 = np.arange(25000)
np.random.shuffle(indice1)
data1 = data1[indice1]
labels1 = labels1[indice1]

indice2 = np.arange(25000)
np.random.shuffle(indice2)
data2 = data2[indice2]
labels2 = labels2[indice2]

embeddings_index = {}
wordX = np.load('/media/yan/winD/docker/word2vec/words/word2vec.npy')
f = codecs.open('/media/yan/winD/docker/word2vec/words/wordsInWord2vec.pkl', 'rb')
allwords = pickle.load(f)
f.close()
for i in range(3000000):
    embeddings_index[''.join(allwords[i])] = wordX[i, :]
# f=codecs.open('/media/yan/winD/docker/glove6B/glove.6B.100d.pkl','rb')
# embeddings_index=pickle.load(f)
# f.close()

embedding_matrix = np.zeros((num_words, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None and i < num_words:
        embedding_matrix[i] = embedding_vector

np.save('../temp/Xtrain_cnn.npy', data1)
np.save('../temp/Xtest_cnn.npy', data2)
np.save('../temp/Ytrain.npy', labels1)
np.save('../temp/Ytest.npy', labels2)
np.save('../temp/embeddingMatrix.npy', embedding_matrix)
