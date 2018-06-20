import codecs
import json
from collections import defaultdict
from collections import Counter


path = './data/usagov_bitly_data2012-03-16-1331923249.txt'

records =[json.loads(line) for line in codecs.open(path,encoding='utf-8')]
print (records[0],"++",records[0]['tz'])
