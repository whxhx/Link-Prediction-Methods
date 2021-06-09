# # Weisfeiler-Lehman Neural Machine for Link Prediction
'''
Warning (from warnings module):
  File "E:\Software\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py", line 571
    % self.max_iter, ConvergenceWarning)
ConvergenceWarning: Stochastic Optimizer: Maximum iterations (100) reached and the optimization hasn't converged yet.

https://blog.csdn.net/dontla/article/details/99853290

报错信息就是说迭代了400次但是还是没达到最佳拟合(不设置的话默认是迭代200次).

既然这样, 我们增加迭代次数试试, 比如将 max_iter 改成1000次

'''
file_list=[
"./redu/CA-AstroPh-2.mat",
"./redu/CA-CondMat-2.mat",
"./redu/CA-HepPh-2.mat",
"./redu/CA-HepTh-2.mat",
"./redu/caida-2.mat",
"./redu/chess-2.mat",
"./redu/pretty-2.mat",
"./redu/gnutella-2.mat",
"./redu/gplus-2.mat",
"./redu/brightkite-2.mat"
]


import numpy as np
def prime(x):
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True
    for i in range(2, x):
        if x % i == 0:
            return False
    return True

prime_numbers = np.array([i for i in range (100000) if prime(i)], dtype=np.int)
# print(prime_numbers.shape)#(9592,)


#------------------------------------------------------------------------------
import networkx as nx
import scipy.sparse
import matplotlib.pyplot as plt

import scipy.io
import math
import pandas as pd
from sklearn.utils import shuffle
from functools import partial
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from time import time
import traceback,os
from decimal import Decimal 

#最大迭代次数
MY_max_iter=100# 此时的auc就很大 #3000
####file_list=['./data/USAir.mat']
#------------------------------------------------------------------------------
for i in range(3):#重复三次
    for dataset_name in file_list:
        try:
            print("------------"+dataset_name+"-------------\n")
            #####print("# ## Importing the dataset")
            data = scipy.io.loadmat(dataset_name)
            # print(data)
            time_start=time()#-----------------------

            #####print("# ## Create a graph from the adjacency matrix")
            a=data['net'].astype(np.int8)
            a=a.todense()
            network = nx.from_numpy_matrix(a)#--------------------有的矩阵太大了直接在这里爆掉
            # print(dataset_name,"   ",network.number_of_nodes(), network.number_of_edges())
            # continue#````````
        
            print("nodes,edges:",network.number_of_nodes(), network.number_of_edges())
            

            #####print("# ## Dividing the network into the training and test networks")
            network_train = network.copy()
            network_test = nx.empty_graph(network_train.number_of_nodes())

            test_ratio = 0.1
            n_links = network_train.number_of_edges()
            n_links_test = math.ceil(test_ratio * n_links)
            #print("n_links_test:",n_links_test)



            selected_links_id = np.random.choice(np.arange(n_links), size=n_links_test, replace=False)

            network_adj_matrix = nx.adj_matrix(network)
            network_adj_matrix = scipy.sparse.triu(network_adj_matrix, k=1)
            row_index, col_index = network_adj_matrix.nonzero()
            links = [(x, y) for x, y in zip(row_index, col_index)]



            selected_links = []
            for link_id in selected_links_id:
                selected_links.append(links[link_id])
            network_train.remove_edges_from(selected_links)
            network_test.add_edges_from(selected_links)

            #####print("network_train.number_of_edges(), network_test.number_of_edges():",network_train.number_of_edges(), network_test.number_of_edges())


            #####print("# ## Sampling negative links")
            k = 2
            n_links_train_pos = network_train.number_of_edges()
            n_links_test_pos = network_test.number_of_edges()
            n_links_train_neg = k * n_links_train_pos
            n_links_test_neg = k * n_links_test_pos


            neg_network = nx.empty_graph(network.number_of_nodes())
            links_neg = list(nx.non_edges(network))
            neg_network.add_edges_from(links_neg)


            n_links_neg = neg_network.number_of_edges()
            ######print("n_links_neg:",n_links_neg)


            selected_links_neg_id = np.random.choice(np.arange(n_links_neg), size=n_links_train_neg + n_links_test_neg, replace=False)

            neg_network_train = nx.empty_graph(network.number_of_nodes())
            neg_network_test = nx.empty_graph(network.number_of_nodes())


            selected_links = []
            for i in range(n_links_train_neg):
                link_id = selected_links_neg_id[i]
                selected_links.append(links_neg[link_id])
            neg_network_train.add_edges_from(selected_links)

            selected_links = []
            for i in range(n_links_train_neg, n_links_train_neg + n_links_test_neg):
                link_id = selected_links_neg_id[i]
                selected_links.append(links_neg[link_id])
            neg_network_test.add_edges_from(selected_links)


            ######print("neg_network_train.number_of_nodes(), neg_network_test.number_of_nodes():",neg_network_train.number_of_nodes(), neg_network_test.number_of_nodes())
            ######print("neg_network_train.number_of_edges(), neg_network_test.number_of_edges():",neg_network_train.number_of_edges(), neg_network_test.number_of_edges())



            #####print("# ## Grouping training and test links")


            all_links_train = list(network_train.edges) + list(neg_network_train.edges)
            label_train = [1] * len(network_train.edges) + [0] * len(neg_network_train.edges)


            all_links_test = list(network_test.edges) + list(neg_network_test.edges)
            label_test = [1] * len(network_test.edges) + [0] * len(neg_network_test.edges)


            y_train, y_test = np.array(label_train), np.array(label_test)


            print("# ## Extracting enclosing subgraph for each links")

            link = all_links_train[12]
            ######print("link:",link)



            fringe = [link]
            subgraph = nx.Graph()



            def enclosing_subgraph(fringe, network, subgraph, distance):
                neighbor_links = []
                for link in fringe:
                    u = link[0]
                    v = link[1]
                    neighbor_links = neighbor_links + list(network.edges(u))
                    neighbor_links = neighbor_links + list(network.edges(v))
                tmp_subgraph = subgraph.copy()
                tmp_subgraph.add_edges_from(neighbor_links)
                # Remove duplicate and existed edge
                neighbor_links = [li for li in tmp_subgraph.edges() if li not in subgraph.edges()]
                tmp_subgraph = subgraph.copy()
                tmp_subgraph.add_edges_from(neighbor_links, distance=distance, inverse_distance=1/distance)
                return neighbor_links, tmp_subgraph


            fringe, subgraph = enclosing_subgraph(fringe, network_train, subgraph, distance=1)
            #####nx.draw(subgraph, with_labels=True);


            def extract_enclosing_subgraph(link, network, size=10):
                fringe = [link]
                subgraph = nx.Graph()
                distance = 0
                subgraph.add_edge(link[0], link[1], distance=distance)
                while subgraph.number_of_nodes() < size and len(fringe) > 0:
                    distance += 1
                    fringe, subgraph = enclosing_subgraph(fringe, network, subgraph, distance)
                
                tmp_subgraph = network.subgraph(subgraph.nodes)
                additional_edges = [li for li in tmp_subgraph.edges if li not in subgraph.edges]
                subgraph.add_edges_from(additional_edges, distance=distance+1, inverse_distance=1/(distance+1))
                return subgraph



            e_subgraph = extract_enclosing_subgraph(link, network_train)
            # nx.draw(e_subgraph, with_labels=True)

            '''
            #####print("e_subgraph[6]:",e_subgraph[6])

            get_ipython().run_cell_magic('timeit', '', 'extract_enclosing_subgraph(link, network_train)')

            get_ipython().run_cell_magic('timeit', '', 'for link in all_links_train:\n    e_subgraph = extract_enclosing_subgraph(link, network_train)')
            '''

            #####print("# ## Subgraph encoding")
            # ### Palette-WL for vertex ordering

            def compute_geometric_mean_distance(subgraph, link):
                u = link[0]
                v = link[1]
                subgraph.remove_edge(u, v)
                
                n_nodes = subgraph.number_of_nodes()
                u_reachable = nx.descendants(subgraph, source=u)
                v_reachable = nx.descendants(subgraph, source=v)
            #     #####print(u_reachable, v_reachable)
                for node in subgraph.nodes:
                    distance_to_u = 0
                    distance_to_v = 0
                    if node != u:
                        distance_to_u = nx.shortest_path_length(subgraph, source=node, target=u) if node in u_reachable else 2 ** n_nodes
                    if node != v:
                        distance_to_v = nx.shortest_path_length(subgraph, source=node, target=v) if node in v_reachable else 2 ** n_nodes
                    #---------------------------------------------------Decimal
                    subgraph.nodes[node]['avg_dist'] = float(Decimal(math.sqrt(Decimal(distance_to_u * distance_to_v))))
                
                subgraph.add_edge(u, v, distance=0)
                
                return subgraph


            e_subgraph = compute_geometric_mean_distance(e_subgraph, link)

            avg_dist = nx.get_node_attributes(e_subgraph, 'avg_dist')


            def palette_wl(subgraph, link):
                tmp_subgraph = subgraph.copy()
                if tmp_subgraph.has_edge(link[0], link[1]):
                    tmp_subgraph.remove_edge(link[0], link[1])
                avg_dist = nx.get_node_attributes(tmp_subgraph, 'avg_dist')
                
                df = pd.DataFrame.from_dict(avg_dist, orient='index', columns=['hash_value'])
                df = df.sort_index()
                df['order'] = df['hash_value'].rank(axis=0, method='min').astype(np.int)
                df['previous_order'] = np.zeros(df.shape[0], dtype=np.int)
                adj_matrix = nx.adj_matrix(tmp_subgraph, nodelist=sorted(tmp_subgraph.nodes)).todense()
                while any(df.order != df.previous_order):
                    df['log_prime'] = np.log(prime_numbers[df['order'].values])
                    total_log_primes = np.ceil(np.sum(df.log_prime.values))
                    df['hash_value'] = adj_matrix * df.log_prime.values.reshape(-1, 1) / total_log_primes + df.order.values.reshape(-1, 1)
                    df.previous_order = df.order
                    df.order = df.hash_value.rank(axis=0, method='min').astype(np.int)
                nodelist = df.order.sort_values().index.values
                return nodelist



            nodelist = palette_wl(e_subgraph, link)
            #nodelist

            size = 10
            if len(nodelist) > size:
                nodelist = nodelist[:size]
                e_subgraph = e_subgraph.subgraph(nodelist)
                nodelist = palette_wl(e_subgraph, link)
            #nodelist

            ######nx.draw(e_subgraph, with_labels=True);


            #e_subgraph.nodes[7]


            #####print("# ### Represent enclosing subgraphs as adjacency matrices")

            def sample(subgraph, nodelist, weight='weight', size=10):
                adj_matrix = nx.adj_matrix(subgraph, weight=weight, nodelist=nodelist).todense()
                vector = np.asarray(adj_matrix)[np.triu_indices(len(adj_matrix), k=1)]
                d = size * (size - 1) // 2
                if len(vector) < d:
                    vector = np.append(vector, np.zeros(d - len(vector)))
                return vector[1:]


            sample(e_subgraph, nodelist, size=10)

            #####print("# ### Subgraph encoding test")

            #link

            e_subgraph = extract_enclosing_subgraph(link, network_train, size=10)
            e_subgraph = compute_geometric_mean_distance(e_subgraph, link)
            nodelist = palette_wl(e_subgraph, link)
            if len(nodelist) > size:
                nodelist = nodelist[:size]
                e_subgraph = e_subgraph.subgraph(nodelist)
                nodelist = palette_wl(e_subgraph, link)
            embeded = sample(e_subgraph, nodelist, size=10)

            ######print("embeded:",embeded)


            #####print("# ## Enclosing subgraph encoding for each links")

            def encode_link(link, network, weight='weight', size=10):
                e_subgraph = extract_enclosing_subgraph(link, network, size=size)
                e_subgraph = compute_geometric_mean_distance(e_subgraph, link)
                nodelist = palette_wl(e_subgraph, link)
                if len(nodelist) > size:
                    nodelist = nodelist[:size]
                    e_subgraph = e_subgraph.subgraph(nodelist)
                    nodelist = palette_wl(e_subgraph, link)
                embeded_link = sample(e_subgraph, nodelist, weight=weight, size=size)
                return embeded_link


            # from functools import partial

         
            X_train = np.array(list(map(partial(encode_link, network=network_train, weight='weight', size=10), all_links_train)))


            ######print("X_train.shape:",X_train.shape)

            X_test = np.array(list(map(partial(encode_link, network=network_train, weight='weight', size=10), all_links_test)))


            # from sklearn.utils import shuffle
            X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train)

            #X_train_shuffle.shape, y_train.shape


            print("------------------# ## Neural Network Learning--------------------")

            # from sklearn.neural_network import MLPClassifier

            model = MLPClassifier(hidden_layer_sizes=(32, 32, 16),
                                alpha=1e-3,
                                batch_size=128,
                                learning_rate_init=0.001,
                                max_iter=MY_max_iter,
                                verbose=True,
                                early_stopping=False,
                                tol=-10000)


            model.fit(X_train_shuffle, y_train_shuffle)
            predictions = model.predict(X_test)


            # from sklearn import metrics
            ##from sklearn.metrics import f1_score

            fpr, tpr, thresholds = metrics.roc_curve(label_test, predictions, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            # print("----------------------------------")# print("auc:")
            #print(auc)
            time_end=time()#单位：s
            with open("res-main-v2.txt",'a',encoding='utf-8') as f1:
                f1.write(dataset_name[7:]+"  "+str(auc)+"  "+str(time_end-time_start)+'\n')

            '''
            print("f1_score:")
            print(f1_score(label_test, predictions, average='macro'))  # 0.26666666666666666
            print(f1_score(label_test, predictions, average='micro'))  # 0.3333333333333333
            print(f1_score(label_test, predictions, average='weighted'))  # 0.26666666666666666
            print(f1_score(label_test, predictions, average=None))  # [0.8 0.  0. ]
            print("----------------------------------")
            '''
        except Exception as e:
            with open(dataset_name[7:-4]+'.log.txt','a',encoding='utf-8') as f:
                f.write(dataset_name[7:]+':\n'+str(e)+'\n'+str(traceback.format_exc())+'\n\n')
        print(dataset_name+'OK!')
    print("loop  "+str(i)+"  OK!!!")
print("-----ALL OK!!!----")
