import numpy as np
from sklearn.metrics import roc_auc_score
import os
import xlrd
import time
import networkx as nx
import random

def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


def HrForNode2vec(category, dataname, dim, klist):
    # read representations of nodes
    file = open('./testemb/' + dataname + '.emb')
    dim1 = dim2 = 0
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        break
    matrix = np.zeros((dim, 128), dtype=float)
    for line in file:
        line = list(map(float, line.strip().split(' ')))
        matrix[int(line[0]), :] = line[1::]
    file.close()
    G = nx.read_edgelist('./finaldata/'+category+'/'+dataname+'.txt', nodetype=int)
    choosenum = int(nx.number_of_nodes(G)*0.1)
    file = open('./finaldata/' + category + '/' + dataname + '_pos.txt')
    hitratio = [0, 0, 0]
    testnum = 0
    mrr = 0
    for line in file:
        line = list(map(int, line.strip().split('\t')))
        sim = cosin_distance(matrix[line[0]], matrix[line[1]])
        canfind = list(nx.non_neighbors(G, line[0]))
        calscr=[]
        for i in range(0, choosenum):
            calscr.append(cosin_distance(matrix[line[0]], matrix[random.choice(canfind)]))
        calscr.append(sim)
        calscr.sort(reverse=True)
        hitindex = calscr.index(sim) + 1
        mrr = mrr + 1/float(hitindex)
        for i in range(0, 3):
            if hitindex <= klist[i]:
                hitratio[i] = hitratio[i] + 1
        testnum = testnum + 1
    output = open('hitratio.txt', 'a')
    output.write(dataname + ' ' + str(float(hitratio[0])/float(testnum)) + ' ' + str(float(hitratio[1])/float(testnum))
                 + ' ' + str(float(hitratio[2])/float(testnum)) + ' ' + str(mrr/testnum)+'\n')
    print float(hitratio[0])/float(testnum)
    print float(hitratio[1])/float(testnum)
    print float(hitratio[2])/float(testnum)
    print mrr/testnum
    file.close()


def MAPforembedding(category, dataname, dim):
    dim = dim+1
    file = open('./testemb/' + dataname + '.emb')
    dim1 = dim2 = 0
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        break
    matrix = np.zeros((dim, 128), dtype=float)
    for line in file:
        line = list(map(float, line.strip().split(' ')))
        matrix[int(line[0]), :] = line[1::]
    file.close()
    G = nx.read_edgelist('./dividedata/'+category+'/'+dataname+'.txt', nodetype=int)
    file = open('./dividedata/' + category + '/' + dataname + '_pos.txt')
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        G.add_edge(int(line[0]), int(line[1]))
    file.close()
    file = open('./dividedata/' + category + '/' + dataname + '_pos.txt')
    Map = 0
    nodesize = 0
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        query = int(line[0])
        nei = nx.neighbors(G, query)
        neiresult = []
        allresult = []
        for node in nei:
            tmp = cosin_distance(matrix[query], matrix[node])
            neiresult.append(tmp)
            allresult.append(tmp)
        nonei = nx.non_neighbors(G, query)
        while True:
            try:
                tmp = cosin_distance(matrix[query],matrix[next(nonei)])
                allresult.append(tmp)
            except StopIteration:
                break
        neiresult.sort(reverse=True)
        allresult.sort(reverse=True)
        recall = 0
        AveP = 0
        neisum = len(neiresult)
        for i in neiresult:
            preindex = allresult.index(i)+1
            findindex = neiresult.index(i)+1
            deltarecal = (float(findindex)/float(neisum))-recall
            recall = float(findindex)/float(neisum)
            persion = float(findindex)/float(preindex)
            AveP = AveP+persion*deltarecal
        Map = Map+AveP
        query = int(line[1])
        nei = nx.neighbors(G, query)
        neiresult = []
        allresult = []
        for node in nei:
            tmp = cosin_distance(matrix[query], matrix[node])
            neiresult.append(tmp)
            allresult.append(tmp)
        nonei = nx.non_neighbors(G, query)
        while True:
            try:
                tmp = cosin_distance(matrix[query],matrix[next(nonei)])
                allresult.append(tmp)
            except StopIteration:
                break
        neiresult.sort(reverse=True)
        allresult.sort(reverse=True)
        recall = 0
        AveP = 0
        neisum = len(neiresult)
        for i in neiresult:
            preindex = allresult.index(i)+1
            findindex = neiresult.index(i)+1
            deltarecal = (float(findindex)/float(neisum))-recall
            recall = float(findindex)/float(neisum)
            persion = float(findindex)/float(preindex)
            AveP = AveP+persion*deltarecal
        Map = Map+AveP
        nodesize = nodesize+2
    Map = Map/nodesize


MAPforembedding('humanreal', 'residence', 241)
