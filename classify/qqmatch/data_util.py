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

data_path = ''
all_ = pd.read_excel('./questionForTrain.xlsx', header=None)

st = pd.read_excel('./data/stopwords.xlsx', header=None)