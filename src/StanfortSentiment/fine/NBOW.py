import codecs
import pickle
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D, SpatialDropout1D

num_words=10000
max_len=50
data_dir1='../../../temp/StanfortSentiment/fine/get_data/'
data_dir2='../../../temp/StanfortSentiment/fine/get_train/'


def get_input():

    with open(data_dir1+'texts.pkl', 'rb') as f:
        text_test = pickle.load(f)[8544:10754]
    with open(data_dir2+'train_sentence.pkl', 'rb') as f:
        text_train = pickle.load(f)
    y_train = np.load(data_dir2+'Ytrain.npy')
    y_test = np.load(data_dir1+'Ytest.npy')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text_train)

    sequence_train = tokenizer.texts_to_sequences(text_train)
    sequence_test= tokenizer.texts_to_sequences(text_test)

    word_index = tokenizer.word_index
    embeddings_index = {}
    wordX = np.load(open("/home/yan/my_datasets/glove/embedding.300d.npy", mode='rb'))
    allwords = pickle.load(open("/home/yan/my_datasets/glove/words.pkl", mode='rb'))
    for i in range(len(allwords)):
        embeddings_index[allwords[i]] = wordX[i, :]
    embedding_matrix = np.zeros((num_words, 300))
    not_fund=0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            not_fund+=1
        if embedding_vector is not None and i < num_words:
            embedding_matrix[i] = embedding_vector
    print(not_fund,'words not found.')
    # sequences = []
    # for i in range(10754):
    #     t = []
    #     tokens = texts[i].lower().split(' ')
    #     for j in range(len(tokens)):
    #         index = word_index.get(tokens[j], 0)
    #         if index < num_words:
    #             t.append(index)
    #         else:
    #             t.append(0)
    #     sequences.append(t)

    print('Found %s unique tokens.' % len(word_index))

    data1 = pad_sequences(sequence_train, maxlen=max_len)
    data2 = pad_sequences(sequence_test, maxlen=max_len)

    indice1 = np.arange(len(data1))
    np.random.shuffle(indice1)
    x_train = data1[indice1]
    y_train = y_train[indice1]

    indice2 = np.arange(len(data2))
    np.random.shuffle(indice2)
    x_test = data2[indice2]
    y_test = y_test[indice2]

    return x_train,y_train,x_test,y_test,embedding_matrix

def get_model(embedding_matrix):
    main_input = Input(shape=(max_len,))
    # init_method = keras.initializers.Orthogonal()
    #embedding1 = Embedding(num_words, 300, weights=[embedding_matrix])(main_input)
    embedding1 = Embedding(num_words, 300)(main_input)
    a = SpatialDropout1D(0.3)(embedding1)
    # x=Dropout(rate=0.3)(embedding1)
    x = GlobalAveragePooling1D()(a)
    # x = Dense(300, activation='relu')(x)
    # x = Dense(300, activation='relu')(x)

    init_2 = keras.initializers.RandomUniform(minval=-1, maxval=1)
    # embedding2 = Embedding(num_words, 300,embeddings_initializer=init_2)(main_input)
    # b=SpatialDropout1D(0.3)(embedding2)
    # b = GlobalMaxPooling1D()(a)

    # z = keras.layers.concatenate([x, b])
    # z = Dense(300, activation='relu')(z)
    # z = Dense(300, activation='relu')(z)

    output = Dense(5, activation='softmax')(x)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_train,y_train,x_test, y_test, embedding_matrix = get_input()
    model = get_model(embedding_matrix)
    hist=model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test,y_test))