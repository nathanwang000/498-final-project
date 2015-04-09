'''
Author: Jiaxuan Wang
This is a python script to extract non onpage features for 2 step graphs
'''
from bs4 import BeautifulSoup
from urlparse import urlparse
import networkx as nx
import json

def genGraph(page_list):
    '''given a list of the form:
    [{rlabel: 1 for researcher website, 0 for non researcher website, -1 for unlabeled website
      plabel: 1 for personal website, 0 for non personal website, -1 for unlabeled website
      url: the url of the page
      head: the header of the html page
      body: the body of the html page}]
    generate a drected graph of the whole network'''
    G = nx.DiGraph()
    # urls are nodes
    G.add_nodes_from(map(lambda x: x['url'], page_list))
    # add edges between nodes
    for page in page_list:
        # extract links in body tag
        soup = BeautifulSoup(page['body'])
        for link in soup.find_all('a'):
            url = link.get('href')
            # deal with relative url
            if not url.startswith('http') and not url.startswith('https'):
                urljoin(page['url'], url)

            # extract graph
    return G

def neighbor_page(G):
    pass

if __name__ == '__main__':
    page_list = json.load(open('../dataset/final_dataset.json'))
    genGraph(page_list)
