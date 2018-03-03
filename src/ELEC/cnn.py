import numpy as np
import pickle
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input,Embedding, \
    GlobalAveragePooling1D, AveragePooling1D

num_words = 30000
max_len = 1000
word_dimension = 500
data_dir='../../temp/elec/util/'
def get_data():
    with open(data_dir+'texts.pkl','rb') as f:
        texts=pickle.load(f)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(texts[:25000])
    sequences = tokenizer.texts_to_sequences(texts)

    # word_index = tokenizer.word_index
    # sequences = []
    # for i in range(50000):
    #     t = []
    #     tokens = texts[i].lower().split(' ')
    #     for j in range(len(tokens)):
    #         index = word_index.get(tokens[j], 0)
    #         if index < num_words:
    #             t.append(index)
    #         else:
    #             t.append(0)
    #     sequences.append(t)

    x = pad_sequences(sequences, maxlen=max_len)
    x_train = x[:25000]
    x_test = x[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    return x_train, y_train, x_test, y_test


def build_model():
    x_train, y_train, x_test, y_test = get_data()
    main_input = Input(shape=(500,))
    x = Embedding(30000, 500, trainable=True)(main_input)
    # x = Embedding(30000, 300, weights=[embedding_matric], trainable=False)(main_input)
    x = Conv1D(filters=500, kernel_size=3, padding='valid', activation='relu')(x)
    x = GlobalMaxPooling1D(name='pool')(x)
    # x = Dense(250,activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = AveragePooling1D(pool_size=3, strides=1)(x)
    # x = GlobalMaxPooling1D()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[main_input], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    build_model()
