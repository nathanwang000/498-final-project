__author__ = 'Jin Zhang'

import json
import sys
import random

print sys.argv
assert len(sys.argv) >= 2, "not enough input, need filename to annotate"
filein = sys.argv[1]

with open(filein) as data_file:
    data = json.load(data_file)
url_diction = {}
positivecount = 0
personalcount = 0



for item in data:
    url = item['url']
    if url.startswith("https://"):
        item['url'] = url[8:]
    if url.startswith("http://"):
        item['url'] = url[7:]
    if  item['url'] in url_diction:
        data.remove(item)
    else:
        url_diction[item['url']] = 1
        
random.shuffle(data)

with open(filein[:-5]+'_dedupe_and_shuffle.json','wb') as page2:
    json.dump(data,page2)
