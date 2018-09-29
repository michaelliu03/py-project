# -*- coding:utf-8 -*-
import codecs
import xlrd
import xlwt
import pandas as pd
import jieba
import codecs
import sys
import json
import re
import pickle

import  numpy as np

global stopwords
stopwords = []

data_path = '../data/questionForTrain.xlsx'
stop_words_path = '../dict/stop_words.utf8'


# del num
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def replace_num(string):
    if(is_number(string)):
        string = '*'
    return string




# 输出字典
def output_vocabulary(abc):
    values = abc[:]
    keys = abc.index
    vocabulary = dict(zip(keys, values))
    with open('../model/words_index.json','w') as outfile:
        json.dump(vocabulary, outfile, indent=4)

def output_labels(labels):
    with open('../model/labels_index.json', 'w') as outfile:
             json.dump(labels, outfile, indent=4)



def train_test_split(x,y,test_size):
    test_len = int(x.shape[0]*test_size)
    x_test = x[:test_len]
    x_train = x[test_len:]
    y_test = y[:test_len]
    y_train = y[test_len:]
    return x_train, x_test, y_train, y_test

def load_data(maxlen, min_count, test_train_ratio):
    all_ = pd.read_excel(data_path, header=None) # id , text , compute value , 标注
    st = codecs.open(stop_words_path,'r',encoding='utf-8')
    all_[1] = all_[1].apply(lambda s: replace_num(s))
    all_['words'] = all_[1].apply(lambda s: list(jieba.cut(s)))
    print(all_.head(5))
    print(all_['words'].head(5))

    # 原版用的是OneHot，这里用word2vector,首先组织content
    content = []
    for i in all_['words']:
        content.extend(i)

    # 组织字典
    abc = pd.Series(content).value_counts()
    abc = abc[abc >= min_count]
    abc[:] = range(1, len(abc) + 1)
    vocabulary_size = len(abc)
    print('vocabulary_size')
    print(vocabulary_size)
    abc[''] = 0

    output_vocabulary(abc) # 保存字典

    #查字典并word2vector
    def doc2num(s, maxlen):
        s = [i for i in s if i in abc.index]
        s = s[:maxlen - 1] + [''] * max(1, maxlen - len(s))
        return list(abc[s])


    all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))
    idx = range(len(all_))
    np.random.shuffle(idx)
    all_ = all_.loc[idx]
    print(all_.head(5))

    x_train = np.array(list(all_['doc2num']))
    labels = sorted(list(set(all_[2])))
    print(labels)
    output_labels(labels)
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    y_train = np.array(all_[2].apply(lambda y: label_dict[y]).tolist())
#   split dataset for training and validation use
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_train_ratio)
    return (x_train, y_train), (x_test, y_test),vocabulary_size, maxlen


def batch_generator(X, y, batch_size):
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


def load_data_prediction(maxlen, min_count):
    all_ = pd.read_excel('../data/ProductHalfMThirdCat.xlsx', header=None)
    #  all_ = pd.read_excel('./data/ProductNewBrandSecondClass.xlsx', header=None)
    all_[1] = all_[1].apply(lambda s: replace_num(s))
    all_['words'] = all_[1].apply(lambda s: list(jieba.cut(s)))
    words_index = json.loads(open('../model/words_index.json').read())
    labels = json.loads(open('../model/labels_index.json').read())
    abc = pd.Series(words_index)
    print(all_['words'])

    def doc2numpre(s, maxlen):
        s = [i for i in s if i in abc.index]
        s = s[:maxlen - 1] + [''] * max(1, maxlen - len(s))
        return list(abc[s])

    print('one hot conversion')
    all_['doc2num'] = all_['words'].apply(lambda s: doc2numpre(s, maxlen))

    vocabulary_size = len(words_index) - 1
    x_train = np.array(list(all_['doc2num']))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    y_train = np.array(all_[2].apply(lambda y: label_dict[y]).tolist())
    return x_train, y_train, vocabulary_size, maxlen, all_[2], num_labels