import numpy as np
import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Conv1D, GlobalMaxPooling1D,Input,Embedding,GlobalAveragePooling1D
#from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical

num_words=10000
max_len=500


def get_input():
    (x_train,y_train),(x_test,y_test)=keras.datasets.reuters.load_data(path='reuters.npz',
                                                                       num_words=num_words,
                                                                       test_split=0.1,)
    print('average length: ',sum([len(seq) for seq in x_train])/len(x_train))
    x_train=pad_sequences(x_train,maxlen=max_len)
    x_test=pad_sequences(x_test,maxlen=max_len)
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)

    word_index = keras.datasets.reuters.get_word_index(path="reuters_word_index.json")
    print('vocabulary size: ',len(word_index))
    embeddings_index = {}
    wordX = np.load(open("/home/yan/my_datasets/glove/embedding.300d.npy", mode='rb'))
    allwords = pickle.load(open("/home/yan/my_datasets/glove/words.pkl", mode='rb'))
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector

    return x_train,y_train,x_test,y_test,embedding_matrix


def get_model(embedding_matrix):
    model = Sequential()
    #model.add(Embedding(35000,50,input_length=500))
    model.add(Embedding(num_words,300,weights=[embedding_matrix],input_length=max_len,trainable=True))
    model.add(Conv1D(filters=250,kernel_size=3,padding='valid',activation='relu'))
    model.add(GlobalMaxPooling1D(name='pool'))
    # model.add(Dense(250,activation='relu'))
    # model.add(Dropout(0.2))
    #model.add(GlobalAveragePooling1D())
    #model.add(Dropout(0.5))
    model.add(Dense(46,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
    return model

if __name__=='__main__':
    x_train, y_train, x_test, y_test, embedding_matrix=get_input()
    model=get_model(embedding_matrix)
    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))