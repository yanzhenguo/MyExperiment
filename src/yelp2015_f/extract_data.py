# -*- coding: utf-8 -*-'

import csv
import pickle
import numpy as np

data_dir='../../temp/yelp2015_f/extract_data/'
train_text=[]
train_label=[]
reader = csv.reader(open('/home/yan/my_datasets/yelp_review_full_csv/train.csv'))
for row in reader:
    train_label.append(int(row[0])-1)
    train_text.append(' '.join(row[1:]))
    # print(int(row[0]))
    # print(' '.join(row[1:]))

test_text=[]
test_label=[]
reader = csv.reader(open('/home/yan/my_datasets/yelp_review_full_csv/test.csv'))
for row in reader:
    test_label.append(int(row[0])-1)
    test_text.append(' '.join(row[1:]))


with open(data_dir+'texts.pkl','wb') as f:
    pickle.dump(train_text+test_text,f)

np.save(data_dir+'label.npy',np.asarray(train_label+test_label,dtype=np.int))
