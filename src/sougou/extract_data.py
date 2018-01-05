# -*- coding: utf-8 -*-'

import csv
import pickle
import numpy as np


data_dir='../../temp/sougou/extract_data/'
train_text=[]
train_label=[]
with open('/home/yan/my_datasets/sogou_news_csv/train.csv') as f:
    for row in f:
        train_label.append(int(row[1])-1)
        train_text.append(row[4:])
        # print(int(row[0]))
        # print(' '.join(row[1:]))

test_text=[]
test_label=[]
with open('/home/yan/my_datasets/sogou_news_csv/test.csv') as f:
    for row in f:
        test_label.append(int(row[1])-1)
        test_text.append(row[4:])


# with open(data_dir+'texts.pkl','wb') as f:
#     pickle.dump(train_text+test_text,f)
#
# np.save(data_dir+'label.npy',np.asarray(train_label+test_label,dtype=np.int))

print(len(train_label))
print(len(test_label))
