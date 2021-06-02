import os
import networkx as nx
import numpy as  np
def devide(category,dataname,ratio):
    print dataname
    G = nx.read_weighted_edgelist('./data/'+category+'/'+dataname+'.txt', nodetype=int)
    nonit = nx.non_edges(G)
    n = nx.number_of_nodes(G)
    n = n*(n-1)/2
    nonedge = n-nx.number_of_edges(G)
    e = nx.number_of_edges(G)
    e = long(ratio*e)
    count = 0
    nonedgechoose = []
    while(True):
        tmp = np.random.random_integers(0, nonedge)
        if tmp not in nonedgechoose:
            nonedgechoose.append(tmp)
            count = count+1
        if count >= e:
            break
    nonedgechoose.sort()
    count = 0
    G_neg = nx.Graph()
    for i in nonedgechoose:
        while count < i:
            next(nonit)
            count = count+1
        G_neg.add_edge(*next(nonit))
        count = count+1
    it = nx.edges(G)
    n = nx.number_of_edges(G)
    n = long(n*ratio)
    count = 0
    edgechoose = []
    while(True):
        tmp = np.random.random_integers(0, nx.number_of_edges(G))
        if tmp not in edgechoose:
            edgechoose.append(tmp)
            count = count+1
        if count >= n:
            break
    edgechoose.sort()
    G_train = nx.Graph()
    G_pos = nx.Graph()
    count = 0
    index = 0
    print len(edgechoose)
    for edge in it.data(False):
        if index >= len(edgechoose):
            G_train.add_edge(*edge)
            continue
        if count != edgechoose[index]:
            G_train.add_edge(*edge)
            count = count+1
            continue
        G_pos.add_edge(*edge)
        count = count+1
        index = index+1
    G_train = nx.DiGraph(G_train)
    nx.write_edgelist(G_train, 'dividedata/'+category+'/'+dataname+'.txt', data=False)
    nx.write_edgelist(G_pos, 'dividedata/'+category+'/'+dataname+'_pos.txt', data=False)
    nx.write_edgelist(G_neg, 'dividedata/'+category+'/'+dataname+'_neg.txt', data=False)
    print 'end'

categories = ['humanreal', 'coauthorship', 'computer', 'humanonline', 'interaction', 'metabolic', 'test']
# categories = ['infrastructure']
for category in categories:
    for root, dirs, files in os.walk('./data/' + category):
        for file in files:
            dataname = os.path.splitext(file)[0]
            devide(category, dataname, 0.1)
