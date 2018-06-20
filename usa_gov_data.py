import codecs
import json
from collections import defaultdict
from collections import Counter


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
    print(time_zones[:10])
    count = get_counts(time_zones)
    print("时区的数量", count['America/New_York'])
    print("the total num of time_zones", len(time_zones))
    print(top_counts(count,5))
    topcounts = top_counts(count, n=10)
    for item in topcounts:
        print(item)

    topcounts_way = Counter(time_zones)
    print(topcounts_way.most_common(10))


if __name__ == "__main__":
    process()