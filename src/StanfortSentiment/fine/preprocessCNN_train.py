import codecs
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer



#-----------------------------------------------------------------------------------
f = codecs.open('D:\\docker\\StanfordSentiment\\FirstMethod\\texts.pkl','rb')
texts = pickle.load(f)
f.close()


#-----------------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)

print('process 1')
f=codecs.open('D:\\docker\\StanfordSentiment\\FirstMethod\\train_sentence.pkl','rb')
train_sentence=pickle.load(f)
f.close()
print(len(train_sentence))
print('process 2')
sequences = tokenizer.texts_to_sequences(train_sentence)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print('process 3')
Xtrain = pad_sequences(sequences, maxlen=50)


np.save('D:\\docker\\StanfordSentiment\\FirstMethod\\Xtrain.npy',Xtrain)