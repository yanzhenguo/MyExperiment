import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import WordPunctTokenizer
train_size = 6920
test_size = 1821
val_size = 872
fWord = codecs.open('D:\\docker\\FirstPaper\\ThirdMethod\\glove.6B.100d.txt','r','utf-8')

#-----------------------------------------------------------------------------------
f = codecs.open('D:\\docker\\StanfordSentiment\\FourthMethod\\texts.pkl','rb')
texts = pickle.load(f)
f.close()


#-----------------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Xtrain = pad_sequences(sequences[0:train_size], maxlen=50)
Xtest = pad_sequences(sequences[train_size:train_size+test_size], maxlen=50)
Xval = pad_sequences(sequences[train_size+test_size:train_size+test_size+val_size], maxlen=50)


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

np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Xtrain.npy',Xtrain)
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Xval.npy',Xval)
np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\Xtest.npy',Xtest)

np.save('D:\\docker\\StanfordSentiment\\FourthMethod\\embeddingMatrix.npy',embedding_matrix)  #16313