import argparse
from collections import Counter
import pickle
import codecs

min_num=10
max_feature=20
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

with open(root_dir + "temp/imdb/keras_code/utils/texts.pkl", 'rb') as f:
    texts = pickle.load(f)
neg_dict=build_dict(texts[:12500],[1])
pos_dict=build_dict(texts[12500:25000],[1])

correct_number=0
predict=[]

for doc in texts[25000:37500]:
    probability=[]
    for token in list(set(doc.split(' '))):
        if neg_dict[token]<min_num or pos_dict[token]<min_num:
            continue
        r = pos_dict[token] / (pos_dict[token] + neg_dict[token])
        probability.append(r)
    probability=sorted(probability)
    result=sum(probability[:30]+probability[-30:])/60
    predict.append(result)
    if result<0.5:
        correct_number+=1

for doc in texts[37500:]:
    probability=[]
    for token in list(set(doc.split(' '))):
        if neg_dict[token]<min_num or pos_dict[token]<min_num:
            continue
        r = pos_dict[token] / (pos_dict[token] + neg_dict[token])
        probability.append(r)

    probability = sorted(probability)
    result = sum(probability[:30]+probability[-30:]) / 60
    predict.append(result)
    if result>0.5:
        correct_number+=1
print(correct_number)
print(correct_number/25000)
#print(predict[:200])
