# coding=utf-8
import os
import os.path
import codecs
import nltk
import pickle
import logging
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def get_text():
    f = codecs.open('../temp/texts.pkl', 'rb')
    texts = pickle.load(f)
    f.close()
    return texts

def adddocument(path):
    '''读取文件中的内容，分词后返回内容'''
    f = codecs.open(path, 'r', 'utf-8')
    paragraph = f.read().lower()
    f.close()
    words = WordPunctTokenizer().tokenize(paragraph)
    return ' '.join(words)


def extract_data():
    '''提取训练文档和测试文档'''
    rootdir = '../../../data/imdb/'
    texts = []
    subdirs = ['train/neg', 'train/pos', 'test/neg', 'test/pos']
    wordTokenizer = WordPunctTokenizer()
    index = 0
    for subdir in subdirs:
        for parent, dirnames, filenames in os.walk(rootdir + subdir):
            for filename in filenames:
                content = adddocument(parent + '/' + filename)
                # 将文档转换为小写，并分词处理
                # content=content.lower()
                # texts.append(" ".join(wordTokenizer.tokenize(content)))
                # 不对文档进行处理，直接保存其原始形式
                texts.append(content)
                index += 1

    out = codecs.open('../../../temp/imdb/keras_code/utils/texts.pkl', 'wb')
    pickle.dump(texts, out, 1)
    out.close()
    print('have processed ', index, ' documents')


def extract_data_un():
    # 提取无标签数据
    rootdir = '../../../data/imdb/'
    texts = []
    subdirs = ['train/unsup']
    for subdir in subdirs:
        for parent, dirnames, filenames in os.walk(rootdir + subdir):
            for filename in filenames:
                content = adddocument(parent + '/' + filename)
                texts.append(content)
    out = codecs.open('../../../temp/imdb/keras_code/utils/texts_unsup.pkl', 'wb')
    pickle.dump(texts, out, 1)
    out.close()


def tokonize_sentence(string):
    # 将文档切分为句子
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return sent_tokenizer.tokenize(string)


def shuffleData(data):
    indice1 = np.arange(len(data[0]))
    np.random.shuffle(indice1)
    data[0] = data[0][indice1]
    data[1] = data[1][indice1]
    indice1 = np.arange(len(data[2]))
    np.random.shuffle(indice1)
    data[2] = data[2][indice1]
    data[3] = data[3][indice1]
    return data



def train_word2vec(inp, outp):
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    import multiprocessing
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    model = Word2Vec(LineSentence(inp), size=1000, window=5, min_count=10,
                     workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    # model.save(outp1)
    model.wv.save_word2vec_format(outp, binary=False)


class ImdbCorpus():
    '''convert imdb documents into inputs of networks'''

    def __init__(self, num_words, max_len, filters='',truncing='post',padding='pre'):
        self.num_words = num_words
        self.max_len = max_len
        self.truncing=truncing
        self.padding=padding
        f = codecs.open('../../../temp/imdb/keras_code/utils/texts.pkl', 'rb')
        self.texts = pickle.load(f)
        f.close()
        tokenizer = Tokenizer(num_words=num_words)
        if filters is not None:
            tokenizer.filters = filters
        tokenizer.fit_on_texts(self.texts[:25000])
        self.tokenizer = tokenizer

    def get_sequence(self):
        return self.tokenizer.texts_to_sequences(self.texts)

    def get_matrix(self):
        return self.tokenizer.texts_to_matrix(self.texts)

    def get_input_bow(self):
        text = self.texts
        xtrain = self.tokenizer.texts_to_matrix(text[:25000])
        xtest = self.tokenizer.texts_to_matrix(text[25000:])
        ytrain = np.zeros((25000,), dtype=np.int8)
        ytest = np.zeros((25000,), dtype=np.int8)
        ytrain[12500:25000] = np.ones((12500,), dtype=np.int8)
        ytest[12500:25000] = np.ones((12500,), dtype=np.int8)
        return [xtrain, ytrain, xtest, ytest]

    def get_sequence_pad(self):
        word_index = self.tokenizer.word_index
        sequences = []
        for i in range(50000):
            t = []
            tokens = self.texts[i].lower().split(' ')
            for j in range(len(tokens)):
                index = word_index.get(tokens[j], 0)
                if index < self.num_words:
                    t.append(index)
                else:
                    t.append(0)
            sequences.append(t)
        return sequences

    def get_input(self):
        sequence = self.get_sequence()
        x=pad_sequences(sequence,
                        maxlen=self.max_len,
                        truncating=self.truncing,
                        padding=self.padding,
                        dtype=np.int32)
        xtrain = x[:25000]
        xtest = x[25000:]
        ytrain = np.zeros((25000,), dtype=np.int64)
        ytest = np.zeros((25000,), dtype=np.int64)
        ytrain[12500:25000] = np.ones((12500,), dtype=np.int64)
        ytest[12500:25000] = np.ones((12500,), dtype=np.int64)
        return xtrain, ytrain, xtest, ytest

    def get_word2vec_matrix(self):
        word_index = self.tokenizer.word_index
        embeddings_index = {}
        wordX = np.load('/home/yan/my_datasets/word2vec/word2vec.npy')
        f = codecs.open('/home/yan/my_datasets/word2vec/words.pkl', 'rb')
        allwords = pickle.load(f)
        f.close()
        for i in range(3000000):
            embeddings_index[''.join(allwords[i])] = wordX[i, :]
        embedding_matrix = np.zeros((self.num_words, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None and i < self.num_words:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


if __name__=='__main__':
    extract_data()