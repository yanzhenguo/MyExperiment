from collections import Counter
from operator import itemgetter

import numpy as np
from nltk.tokenize import WordPunctTokenizer
import pickle
import keras
from keras.layers import Input, Dense
from keras.models import Model
import get_data

root_dir = '../../../'
document_length=500
num_words=20000

def tokenize(sentence, grams):
    words = sentence.split(' ')
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i + gram])]
    return tokens


def build_dict(f, grams):
    dic = Counter()
    for sentence in f:
        dic.update(tokenize(sentence, grams))
    return dic

def compute_ratio(poscounts, negcounts, totalcounts,alpha=1):
    alltokens = totalcounts.keys()
    dic = dict((t, i) for i, t in enumerate(totalcounts))
    d = len(dic)
    print("computing r...")
    p, q = np.ones(d) * alpha, np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()


    r_pos = np.log(p / q)
    r_pos_p=(r_pos/(np.abs(r_pos).max())+1)/2

    r_neg=np.log(q/p)
    r_neg_p = (r_neg / (np.abs(r_neg).max()) + 1) / 2

    r=r_pos
    return dic, r,  r_pos_p, r_neg_p

def main(ngram):
    f = open(root_dir+'temp/imdb/keras_code/utils/texts.pkl', 'rb')
    texts = pickle.load(f)
    f.close()
    # 分词
    wordTokenizer = WordPunctTokenizer()
    for i,doc in enumerate(texts):
        doc=' '.join(wordTokenizer.tokenize(doc))
        texts[i]=doc

    ngram = [int(i) for i in ngram]
    print("counting...")

    poscounts = build_dict(texts[0:12500], ngram)
    negcounts = build_dict(texts[12500:25000], ngram)
    totalcounts = build_dict(texts,ngram)
    totalcounts=dict(sorted(totalcounts.items(),key=itemgetter(1),reverse=1)[:num_words])

    dic, r, r_pos_p, r_neg_p = compute_ratio(poscounts, negcounts,totalcounts)
    print(r[:100])
    model=get_model()
    for i in range(10):
        x_train, y_train, x_test, y_test = get_input(r, r_pos_p, r_neg_p, texts, totalcounts, dic)
        model.fit([x_train], y_train, batch_size=32, epochs=3, validation_data=([x_test], y_test))


def get_input(r, r_pos_p, r_neg_p, texts, totalcounts, dic):
    print('adding weighting to input ...')
    x = np.zeros((50000, num_words), dtype=np.int32)

    rand_num = np.random.rand(25000, num_words)

    for i,doc in enumerate(texts):
        words=doc.split(' ')
        for word in words:
            # if i<12500 and word in totalcounts and rand_num[i,dic[word]]<r_pos_p[dic[word]]:
            if i < 12500 and word in totalcounts:
                x[i,dic[word]]=r[dic[word]]
            # elif i<25000 and word in totalcounts and rand_num[i,dic[word]]<r_neg_p[dic[word]]:
            elif i < 25000 and word in totalcounts:
                x[i, dic[word]] = r[dic[word]]
            elif i>=25000 and word in totalcounts:
                x[i, dic[word]] = r[dic[word]]
            # if i<12500 or (i>=25000 and i<37500):
            #     x[i, dic[word]] = r_pos[dic[word]]
            # else:
            #     x[i, dic[word]] = r_neg[dic[word]]

    x_train=x[:25000]
    x_test=x[25000:]
    y_train = np.zeros((25000,), dtype=np.float32)
    y_test = np.zeros((25000,), dtype=np.float32)
    y_train[12500:25000] = np.ones((12500,), dtype=np.float32)
    y_test[12500:25000] = np.ones((12500,), dtype=np.float32)

    indice = np.arange(25000)
    np.random.shuffle(indice)
    x_train = x_train[indice]
    x_test = x_test[indice]
    y_train = y_train[indice]
    y_test = y_test[indice]

    return x_train, y_train, x_test, y_test

def get_model():
    input_1=Input(shape=(num_words,))
    # x=Dense(500,activation='relu')(input_1)
    output_1=Dense(1,
                   activation='sigmoid',
                   kernel_initializer=keras.initializers.Constant(0.1),
                   bias_initializer=keras.initializers.Constant(0))(input_1)

    model=Model(inputs=input_1,outputs=output_1)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model



if __name__ =='__main__':
    main('1')