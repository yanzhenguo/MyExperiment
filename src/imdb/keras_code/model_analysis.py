# -*- coding: utf-8 -*-
import codecs
import pickle
import heapq
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import WordPunctTokenizer


# num_words = 30000
# num_train = 25000
# f = codecs.open('../temp/texts.pkl', 'rb')
# texts = pickle.load(f)
# f.close()
# tokenizer = Tokenizer(num_words=num_words)
# tokenizer.filters = ''
# tokenizer.fit_on_texts(texts[:num_train])
# sequences = tokenizer.texts_to_sequences(texts)




# analysis weights of a saved model
def preserve_index_word():
    num_words = 30000
    max_len = 500
    num_train = 25000
    num_test = 25000
    f = codecs.open('../temp/texts.pkl', 'rb')
    texts = pickle.load(f)
    f.close()
    newText = []
    for sentence in texts:
        t = []
        words = WordPunctTokenizer().tokenize(sentence)
        for word in words:
            if word.isalpha():
                t.append(word)
        newText.append(' '.join(t))

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.filters = ''
    tokenizer.fit_on_texts(newText[:num_train])
    sequences = tokenizer.texts_to_sequences(newText)
    word_index = tokenizer.word_index
    index_word = {}
    for word, index in word_index.items():
        index_word[index] = word
    # f = codecs.open('../temp/index_word.pkl', 'wb')
    # pickle.dump(index_word, f, 1)
    # f.close()
    return index_word,sequences


def load_index_word():
    f = codecs.open('../temp/index_word.pkl', 'rb')
    index_word = pickle.load(f)
    f.close()
    return index_word


def analysis_embedding_layer(model):
    # index_word = load_index_word()
    embedding_weights = model.get_layer('embedding_1').get_weights()[0]
    # print(type(embedding_weights))
    # for i in range(len(embedding_weights)):
    #     print(embedding_weights[i].shape)
    feature1 = embedding_weights[0]
    print(feature1[:100])
    # maxids = heapq.nlargest(10, range(len(feature1)), feature1.take)
    # print([index_word[id] for id in maxids])


def analysis_dense_layer(model):
    # index_word = load_index_word()
    dense_weights = model.get_layer('dense_1').get_weights()
    print(dense_weights[0][:100])

model = load_model('../temp/bowCNN_same.model.h5')
index,sequence = preserve_index_word()
dense_weights = model.get_layer('dense_1').get_weights()[0]
dense_bias = model.get_layer('dense_1').get_weights()[1]
embedding_weights = model.get_layer('embedding_1').get_weights()[0]
# print(dense_weights[:50])
# print(dense_bias)
tEmbed = embedding_weights[sequence[0]]
nEmbed =np.zeros(shape=(len(sequence[0])-2,800))
for i in range(len(sequence[0])-2):
    nEmbed[i]=np.average(tEmbed[i:i+3],axis=0)
max_index=np.argmax(nEmbed,axis=0)
print(max_index.shape)
for ind,word in enumerate(max_index):
    print(index[sequence[0][word]],end=" ")
    print(index[sequence[0][word+1]], end=" ")
    print(index[sequence[0][word+2]], end=" ")

    if dense_weights[ind]>0:
        print('pos')
    else:
        print('neg')
# print(dense_weights.shape)
# print(dense_bias.shape)
# print(embedding_weights.shape)
