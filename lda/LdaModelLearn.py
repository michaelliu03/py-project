import os
from gensim import corpora
from gensim.models import LdaModel
import codecs

train = []
stopwords = codecs.open('./dict/stop_words.utf8','r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]
fp = codecs.open('./data/query_expansion_seg.csv','r',encoding='utf8')
for line in fp:
    line = line.split()
    train.append([ w for w in line if w not in stopwords ])

dictionary = corpora.Dictionary(train)
corpus = [ dictionary.doc2bow(text) for text in train ]
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)