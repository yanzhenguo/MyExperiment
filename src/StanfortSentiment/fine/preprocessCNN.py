import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import WordPunctTokenizer

fWord = codecs.open('D:\\docker\\FirstPaper\\ThirdMethod\\glove.6B.100d.txt','r','utf-8')

#-----------------------------------------------------------------------------------
f = codecs.open('D:\\docker\\StanfordSentiment\\FirstMethod\\texts.pkl','rb')
texts = pickle.load(f)
f.close()


#-----------------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Xtrain = pad_sequences(sequences[0:8544], maxlen=50)
Xtest = pad_sequences(sequences[8544:10754], maxlen=50)
Xval = pad_sequences(sequences[10754:11855], maxlen=50)


embeddings_index = {}
for line in fWord:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Xtrain.npy',Xtrain)
np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Xval.npy',Xval)
np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Xtest.npy',Xtest)

np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\embeddingMatrix.npy',embedding_matrix)