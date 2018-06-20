import codecs
import json
from collections import defaultdict
from collections import Counter
from pandas import  DataFrame,Series
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt



path = './data/usagov_bitly_data2012-03-16-1331923249.txt'

def read_usa_recods():
   records =[json.loads(line) for line in codecs.open(path,encoding='utf-8')]
   #print (records[0],"++",records[0]['tz'])
   time_zone = [rec['tz'] for rec in records if 'tz' in rec ]
   #print(time_zone[:10])
   return records,time_zone

def get_counts(sequence):
   counts ={}
   for x in  sequence:
       if x in counts:
           counts[x] += 1
       else:
           counts[x] = 1

   return counts

def top_counts(count_dict ,n=10):
    value_key_pairs =[(count,tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
   # print(value_key_pairs[-n:])
    return value_key_pairs[-n:]

def process():
    records,time_zones = read_usa_recods()
    #print(time_zones[:10])
    count = get_counts(time_zones)
    #print("时区的数量", count['America/New_York'])
    #print("the total num of time_zones", len(time_zones))
    #print(top_counts(count,5))
    topcounts = top_counts(count, n=10)
    #for item in topcounts:
    #    print(item)

    topcounts_way = Counter(time_zones)
    #print(topcounts_way.most_common(10))

    frame = DataFrame(records)
    #print(frame.head(5))
    tz_counts = frame['tz'].value_counts()
    #print(tz_counts[:10])
    clean_tz = frame['tz'].fillna('Missing')
    clean_tz[clean_tz == ''] = 'UnKnown' ## 数据补齐
    tz_counts =clean_tz.value_counts()
    #print(tz_counts[:10])
    #draw_bar_analysis('barh',tz_counts,10)

    sresult= Series([x.split()[0] for x in frame.a.dropna()])
    #print(sresult.head(5))
    sresult_count_only8 = sresult.value_counts()[:8]
    #print(sresult_count_only8)
    cframe = frame[frame.a.notnull()]
    #print(type(cframe))
    operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
    #print(operating_system[:5])
    by_tz_os = cframe.groupby(['tz', operating_system])
    agg_counts = by_tz_os.size().unstack().fillna(0)
    #print(agg_counts[:10])
    indexer = agg_counts.sum(1).argsort()
    #print(indexer[:10])
    count_subset = agg_counts.take(indexer)[-10:]
    #print(count_subset)
    count_subset.plot(kind='barh', stacked=True)
    plt.show()
    normed_subset = count_subset.div(count_subset.sum(1), axis=0)
    normed_subset.plot(kind='barh', stacked=True)
    plt.show()

def draw_bar_analysis(type,tz_counts,n):
    plt.figure(figsize=(10, 6))
    tz_counts[:n].plot(kind=type, rot=0)
    plt.show()


if __name__ == "__main__":
    process()