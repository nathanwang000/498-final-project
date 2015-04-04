''' 
Author: Jiaxuan Wang
Description:
This is a simple python script for summarization using pagerank
'''
import networkx as nx
import nltk.data
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class tfidfSIM:
    def __init__(self, documents):
        '''documents is a list of documents'''
        tfidf = TfidfVectorizer(decode_error='ignore', stop_words=None)
        self.transformer = tfidf.fit(documents)

    def sim(self, d1, d2):
        d1_vec = np.array(self.transformer.transform([d1]).todense()).ravel()
        d2_vec = np.array(self.transformer.transform([d2]).todense()).ravel()
        return np.dot(d1_vec, d2_vec) / np.linalg.norm(d1_vec) / np.linalg.norm(d2_vec)

def genGraph(text, sim_class=tfidfSIM):
    ''' 
    This function generate a graph for the target text
    text is the text for summarization
    sim is a class for measuring similarity between sentences
    '''
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    G = nx.Graph()
    # sentences in a text is vertices
    sents = tokenizer.tokenize(text)
    G.add_nodes_from(sents)
    # add weighted edge
    sim = sim_class(sents)
    nodes = G.nodes()
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            G.add_edge(nodes[i],nodes[j],weight=sim.sim(nodes[i],nodes[j]))
    return G

def createSummary(G, top=10):
    '''
    Given a graph and top sentences to display, return a summary
    '''
    ranked_sents = sorted(nx.pagerank(G).items(), key=lambda x: x[1], reverse=True)
    summary = map(lambda x: x[0], ranked_sents)[:top]
    return " ".join(summary)

if __name__ == '__main__':
    G = genGraph(open('testfile.txt').read())
    ranked_sents = sorted(nx.pagerank(G).items(), key=lambda x: x[1], reverse=True)
    summary = createSummary(G)
