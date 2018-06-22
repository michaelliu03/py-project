import  numpy as np
import  codecs
from classify.util import *
from sklearn.cross_validation import  train_test_split



def get_data():
    hamdata, hamlabel = get_ham_data(PosPath)
    spamdata,spamlabel = get_spam_data(NegPath)
    # 构成相关的corpus
    corpus_data = hamdata + spamdata   # 语料相加
    corpus_label = hamlabel + spamlabel # 标签相加
    return corpus_data, corpus_label

def prepare_datasets(corpus, labels, test_data_proportion =0.3):
    train_X, test_X ,train_Y,train_Y = train_test_split(corpus,labels,
                                                        test_size= test_data_proportion,
                                                        random_state=0.3)



def process():
    corpus, labels = get_data()  # 获取数据集
    print("总的数据量:", len(labels))
    corpus, labels = remove_empty_docs(corpus, labels)
    label_name_map = ["垃圾邮件", "正常邮件"]

    # 对数据进行划分
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                            labels,
                                                                            test_data_proportion=0.3)


if  __name__ == "__main__":
    process()


