import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model , load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D, SpatialDropout1D, BatchNormalization

f = codecs.open('../temp/texts.pkl', 'rb')
texts = pickle.load(f)
f.close()

f = codecs.open('../temp/imdb_texts.pkl', 'rb')
imdb_texts = pickle.load(f)
f.close()

tokenizer = Tokenizer(num_words=20000)
tokenizer.filters = ''
tokenizer.fit_on_texts(imdb_texts[:25000])
sequences = tokenizer.texts_to_sequences(texts[25000:])
Xtest = pad_sequences(sequences, maxlen=500)
Ytest = np.zeros((25000,), dtype=np.int8)
Ytest[12500:] = np.ones((12500,), dtype=np.int8)
indice2 = np.arange(25000)
np.random.shuffle(indice2)
Xtest = Xtest[indice2]
Ytest = Ytest[indice2]

model=load_model('../temp/model.h5')
print(model.evaluate(Xtest,Ytest,batch_size=32))