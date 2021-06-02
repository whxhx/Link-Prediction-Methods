import numpy as np
import networkx as nx
import random
import shutil
from sklearn.metrics import roc_auc_score
import os
import xlrd
import time
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

def aucForNode2vec(category, dataname, dim, embfile):
    # read representations of nodes
    dim = dim+1
    # file = open(embfile + dataname + '.emb')
    # model = Word2Vec()
    # model = KeyedVectors.load_word2vec_format(file)
    file = open(embfile + dataname + '.emb')
    dim1 = dim2 = 0
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        dim1 = line
        break
    matrix = np.zeros((dim, dim1[1]), dtype=float)
    for line in file:
        line = list(map(float, line.strip().split(' ')))
        matrix[int(line[0]), :] = line[1::]
    file.close()

    results = []
    label = []
    # calculate sims of positive samples
    file = open('./dividedata/' + category + '/' + dataname + '_pos.txt')
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        sim = cosin_distance(matrix[line[0]], matrix[line[1]])
        if sim != None:
            results.append(sim)
            label.append(1)
    file.close()

    # calculate sims of negative samples
    file = open('./dividedata/' + category + '/' + dataname + '_neg.txt')
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        sim = cosin_distance(matrix[line[0]], matrix[line[1]])
        if sim != None:
            results.append(sim)
            label.append(0)
    file.close()
    #calculate AUC
    return roc_auc_score(np.array(label), np.array(results))


def HrForNode2vec(category, dataname, dim, klist):
    # read representations of nodes
    starttime = time.time()
    dim = dim + 1
    file = open('./Struc2Vecemb/' + dataname + '.emb')
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
    choosenum = int(nx.number_of_nodes(G)*0.1)
    file = open('./dividedata/' + category + '/' + dataname + '_pos.txt')
    hitratio = [0, 0, 0]
    testnum = 0
    mrr = 0
    for line in file:
        line = list(map(int, line.strip().split(' ')))
        sim = cosin_distance(matrix[line[0]], matrix[line[1]])
        for t in range(0, 2):
            G.add_node(line[t])
            canfind = list(nx.non_neighbors(G, line[t]))
            calscr=[]
            for i in range(0, choosenum):
                calscr.append(cosin_distance(matrix[line[t]], matrix[random.choice(canfind)]))
            calscr.append(sim)
            calscr.sort(reverse=True)
            hitindex = calscr.index(sim) + 1
            mrr = mrr + 1/float(hitindex)
            for i in range(0, 3):
                if hitindex <= klist[i]:
                    hitratio[i] = hitratio[i] + 1
            testnum = testnum + 1
    endtime = time.time()
    output = open('./hr/' + category + '/' + 'Struc2Vechitratio.txt', 'a')
    output.write(dataname + ' ' + str(float(hitratio[0])/float(testnum)) + ' ' + str(float(hitratio[1])/float(testnum))
                 + ' ' + str(float(hitratio[2])/float(testnum)) + ' ' + str(mrr/testnum) + ' ' + str(endtime-starttime) + '\n')
    file.close()


def MAPforembedding(category, dataname, dim, test_ratio):
    starttime = time.time()
    dim = dim+1
    file = open('./Struc2Vecemb/' + dataname + '.emb')
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
    choosenum = nx.number_of_nodes(G)*test_ratio
    choosenum = int(choosenum)
    count = 0
    nodechoose = []
    while(True):
        tmp = np.random.random_integers(1, nx.number_of_nodes(G))
        if tmp not in nodechoose:
            nodechoose.append(tmp)
            count = count+1
        if count >= choosenum:
            break
    Map = 0
    nodesize = len(nodechoose)
    for nodet in nodechoose:
        query = int(nodet)
        G.add_node(query)
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
    Map = Map/nodesize
    endtime = time.time()
    output = open('./map/' + category + '/' + 'Struc2Vecmap.txt', 'a')
    output.write(dataname + ' ' + str(Map) + ' ' + str(endtime-starttime) + '\n')


def cosin_distance(vector1, vector2):
    # return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
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


def readExcel():
    readbook = xlrd.open_workbook('./statistics.xlsx')
    sheet = readbook.sheet_by_index(0)
    dic = {}
    for row in range(0, sheet.nrows):
        key = sheet.cell(row, 0).value.encode('utf-8')
        dic[key] = int(sheet.cell(row, 1).value)
    return dic


def runNode2vec():
    # categories = ['computer', 'humanonline', 'humanreal', 'infrastructure', 'interaction', 'metabolic', 'coauthorshiip']
    # categories = ['humanreal', 'infrastructure', 'interaction', 'metabolic']
    categories = ['infrastructure']
    # numV = readExcel()
    for category in categories:
        for root, dirs, files in os.walk('./data/' + category):
            for file in files:
                dataname = os.path.splitext(file)[0]
                print (dataname)
                os.system('/home/wyz/anaconda2/bin/python2.7 node2vec/src/main.py --input '
                          'dividedata/' + category+'/'+dataname
                          + '.txt --output emb/' + dataname
                          + '.emb')
                # if aucForNode2vec(category,dataname,numV[dataname])>aucForNode2vec(category,dataname,numV[dataname], './node2vecemb/'):
                #     shutil.copyfile('./emb/'+dataname+'.emb', './node2vecemb/'+dataname+'.emb')


def runStruc2vec():
    # categories = ['computer', 'humanreal', 'infrastructure', 'interaction', 'metabolic', 'coauthorshiip', 'humanonline']
    # categories = ['humanreal', 'infrastructure', 'interaction', 'metabolic']
    categories = ['test']
    # numV = readExcel()
    for category in categories:
        for root, dirs, files in os.walk('./data/' + category):
            for file in files:
                dataname = os.path.splitext(file)[0]
                print (dataname)
                os.system('/home/wyz/anaconda2/bin/python2.7 struc2vec/src/main.py --input '
                          'dividedata/' + category+'/'+dataname
                          + '.txt --output Struc2Vecemb/' + dataname
                          + '.emb --OPT3 true')


def runDeepWalk():
    # categories = ['coauthorship', 'computer', 'humanonline', 'humanreal', 'infrastructure', 'interaction', 'metabolic']
    # for category in categories:
    #     for root, dirs, files in os.walk('./data/' + category):
    #         for file in files:
    #             dataname = os.path.splitext(file)[0]
    #             print dataname
    #             os.system('python C:\\Users\\carina\\PycharmProjects\\deepwalk-master\\deepwalk\\__main__.py --input '
    #                       'C:\\Users\\carina\\PycharmProjects\\deepwalk-master\\example_graphs\\' + dataname
    #                       + '.edgelist --output C:\\Users\\carina\\PycharmProjects\\deepwalk-master\\output\\' + dataname
    #                       + '.emb')
    # categories = ['computer', 'humanonline', 'humanreal', 'infrastructure', 'interaction', 'metabolic', 'test']
    # categories = ['test']
    categories = ['coauthorship']
    for category in categories:
        for root, dirs, files in os.walk('./data/' + category):
            for file in files:
                dataname = os.path.splitext(file)[0]
                print (dataname)
                os.system('/home/wyz/anaconda2/bin/python2.7 ./deepwalk/deepwalk/__main__.py --input '
                          './finaldata/' + category + '/' + dataname
                          + '.txt --output ./deepwalkemb/' + dataname
                          + '.emb')


def runLine():
    # categories = ['coauthorship', 'computer', 'humanonline', 'humanreal', 'infrastructure', 'interaction', 'metabolic']
    # for category in categories:
    #     for root, dirs, files in os.walk('./data/' + category):
    #         for file in files:
    #             dataname = os.path.splitext(file)[0]
    #             print dataname
    #             os.system('python C:\\Users\\carina\\PycharmProjects\\LINE-master\\windows\\line -train' + dataname
    #                       + '.edgelist --output ' + dataname
    #                       + '.emb -binary 0 -size 128 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 10')
    # categories = ['coauthorship', 'computer', 'humanonline', 'humanreal', 'infrastructure', 'interaction', 'metabolic']
    categories = ['LPAC']
    for category in categories:
        for root, dirs, files in os.walk('./data/' + category):
            for file in files:
                dataname = os.path.splitext(file)[0]
                print (dataname)
                os.system('./line -train ./data/LPAC/' + dataname
                          + '.txt --output ./output/' + dataname
                          + '.emb -binary 0 -size 128 -order 2 -negative 5 -samples 0.0001 -rho 0.025 -threads 10')

def runSEAL():
    categories = ['coauthorship', 'computer', 'humanonline']
	#categories = ['metabolic']
    for category in categories:
        for root, dirs, files in os.walk('./data/' + category):
            for file in files:
                dataname = os.path.splitext(file)[0]
                print (dataname)
                start = time.time()
                for i in range(5, 31, 5):
                    os.system('/home/wyz/anaconda2/bin/python2.7 ./SEAL-master/Python/Main.py --data-name ' + dataname
                            + " --max-nodes-per-hop 100 --max-train-num 10000 --test-ratio "+str(float(i)/100))
                print ((time.time() - start) / 1000)

# numV = readExcel()
# categories = ['coauthorship', 'computer', 'humanonline', 'humanreal', 'infrastructure', 'interaction', 'metabolic']
# for category in categories:
#     for root, dirs, files in os.walk('./data/' + category):
#         for file in files:
#             dataname = os.path.splitext(file)[0]
#             print dataname
#             aucForNode2vec(category, dataname, numV[dataname])
# runNode2vec()
# os.system('C:\\Users\\carina\\PycharmProjects\\LINE-master\\windows\\line -train '
#           + 'C:\\Users\\carina\\PycharmProjects\\LINE-master\\windows\\graph\\karate.edgelist -output '
#           + 'C:\\Users\\carina\\PycharmProjects\\LINE-master\\windows\\output\\karate'
#           + '.emb -binary 0 -size 200 -order 2 -negative 5 -samples 1 -rho 0.025 -threads 10')


#def main():
# for i in range(1, 10):
# runNode2vec()
# runDeepWalk()
# runLine()
# runSEAL()
# runStruc2vec()
# res = open('./Line_1AUC.txt', 'w')
numV = readExcel()
#categories = ['humanreal', 'computer', 'infrastructure', 'interaction', 'metabolic', 'coauthorship', 'humanonline']
categories = ['humanreal']
# categories = ['coauthorship', 'humanonline']
# categories = ['coauthorship', 'metabolic', 'humanonline']
for category in categories:
    for root, dirs, files in os.walk('./data/' + category):
        for file in files:
            dataname = os.path.splitext(file)[0]
            # print dataname
            # print dataname, aucForNode2vec(category, dataname, numV[dataname], './Line_1emb/')
            # res.writelines(dataname+' '+str(aucForNode2vec(category, dataname, numV[dataname], './Line_1emb/'))+'\n')
#             print dataname
#             HrForNode2vec(category, dataname, numV[dataname], [1, 5, 10])
            MAPforembedding(category, dataname, numV[dataname], 0.1)
