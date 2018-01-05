import os
import pdb
import numpy as np
import argparse
from collections import Counter
import pickle
import codecs
# generate features of nbsvm

root_dir = '../../../'

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


def process_files(file_pos, file_neg, dic, r, outfn, grams):
    output = []
    for beg_line, f in zip(["1", "-1"], [file_pos, file_neg]):
        for l in f:
            tokens = tokenize(l, grams)
            indexes = []
            for t in tokens:
                try:
                    indexes += [dic[t]]
                except KeyError:
                    pass
            indexes = list(set(indexes))
            indexes.sort()
            line = [beg_line]
            for i in indexes:
                line += ["%i:%f" % (i + 1, r[i])]
            output += [" ".join(line)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(list(poscounts.keys()) + list(negcounts.keys())))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    print("computing r...")
    p, q = np.ones(d) * alpha, np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p / q)
    return dic, r


def main(ngram):
    f = codecs.open(root_dir+'temp/imdb/keras_code/utils/texts.pkl', 'rb')
    texts = pickle.load(f)
    f.close()

    ngram = [int(i) for i in ngram]
    print("counting...")

    poscounts = build_dict(texts[0:12500], ngram)
    negcounts = build_dict(texts[12500:25000], ngram)

    dic, r = compute_ratio(poscounts, negcounts)
    print("processing files...")
    process_files(texts[0:12500], texts[12500:25000], dic, r, root_dir+"temp/imdb/keras_code/nbsvm/train-nbsvm.txt", ngram)
    process_files(texts[25000:37500], texts[37500:50000], dic, r, root_dir+"temp/imdb/keras_code/nbsvm/test-nbsvm.txt", ngram)



main(ngram='1')