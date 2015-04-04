__author__ = 'Yifei'

import json
import webbrowser
import sys

print sys.argv
assert len(sys.argv) >= 2, "not enough input, need filename to annotate"
filein = sys.argv[1]

with open(filein) as data_file:
    data = json.load(data_file)

positivecount = 0
personalcount = 0
totalpages = 0
annotated = []

for item in data:
    url = item['url']
    annotated.append(item)

    totalpages += 1
    if item['rlabel'] == -1 or item['plabel'] == -1:
        if url.startswith('http'):
            webbrowser.open(url)
        else:
            url = 'http://'+url
            webbrowser.open(url)

        var = raw_input('Page '+str(totalpages)+', Is it a personal page? 1--yes  0---no:   ')
        if var == 'q':
            with open(filein, 'wb') as pages:
                json.dump(data, pages)
            with open('annotated.json', 'wb') as pages:
                json.dump(annotated, pages)
            sys.exit('Stop manually anotating. Total crawled: '+str(len(data))+'. Research page: '+str(positivecount)+'. Total annotated: '+str(totalpages)+'. see you next time!')
        elif var == '1' or var == '0':
            item['plabel'] = int(var)
        else:
            while var != '1' and var != '0':
                var = raw_input('Page '+str(totalpages)+', Is it a personal page? 1--yes  0---no:   ')
            item['plabel'] = int(var)

        if not item['plabel']:
            item['rlabel'] = 0
            continue # assumption: researcher pages are personal pages
        
        var = raw_input('Page '+str(totalpages)+', Is it a research page? 1--yes  0---no:   ')
        if var == 'q':
            with open(filein, 'wb') as pages:
                json.dump(data, pages)
            with open('annotated.json', 'wb') as pages:
                json.dump(annotated, pages)
            sys.exit('Stop manually anotating. Total crawled: '+str(len(data))+'. Research page: '+str(positivecount)+'. Total annotated: '+str(totalpages)+'. see you next time!')
        elif var == '1' or var == '0':
            item['rlabel'] = int(var)
        else:
            while var != '1' and var != '0':
                var = raw_input('Page '+str(totalpages)+', Is it a research page? 1--yes  0---no:   ')
            item['rlabel'] = int(var)

    if item['rlabel'] == 1:
        positivecount += 1
    if item['plabel'] == 1:
        personalcount+=1

print 'research pages: '+str(positivecount)+'  non research pages'+str(len(data)-positivecount)
