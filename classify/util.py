import codecs
import numpy as np

PosPath = "../data/classification/ham_data.txt"
NegPath = "../data/classification/spam_data.txt"

def get_ham_data(path):
    ham_data_ = codecs.open(path, 'r', encoding='utf-8')
    ham_data = list(ham_data_)
    ham_label = np.ones(len(ham_data)).tolist()
    return ham_data,ham_label

def get_spam_data(path):
    spam_data_ = codecs.open(path,'r',encoding='utf-8')
    spam_data  = list(spam_data_)
    #print(spam_data)
    spam_label = np.zeros(len(spam_data)).tolist()
    #print(spam_label)
    return spam_data,spam_label

def remove_empty_docs(corpus,labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus,labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    return filtered_corpus,filtered_labels