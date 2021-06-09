from os import truncate
from sys import argv
import networkx as nx
from networkx.algorithms.centrality import eigenvector
from networkx.classes.function import degree
from numpy.core.fromnumeric import sort
import numpy as np
import math
import os

class GraphEval:
    def __init__(self, filePath):
        gra = nx.Graph()
        with open(filePath, "r") as f:
            for line in f:
                i, j = line.split()[0], line.split()[1]
                i, j = int(i), int(j)
                if i != j:
                    gra.add_edge(i, j)
        gra = gra.subgraph(max(nx.connected_components(gra), key=len))
        deglist = []
        for node in gra.nodes():
            deglist.append((node, gra.degree[node]))

        def fun4sort(ele):
            return ele[1]

        deglist.sort(key=fun4sort, reverse=True)
        self.deglist = deglist
        self.gra = gra
        self.number_of_nodes = gra.number_of_nodes()
        self.number_of_edges = gra.number_of_edges()

    def eval(self):
        print("Num of Nodes:", self.number_of_nodes)
        print("Num of Edges:", self.number_of_edges)
        print("Average Degree", self.average_degree())
        print("Clustering Coefficient", self.clustering_coefficient())
        print("Assortativity Coefficient", self.assortativity_coefficient())
        print("Power Law Exponent", self.power_law_exponent())
        print("Edge Distribution Entropy", self.edge_distrubution_entropy())
        print("Algebraic Connectivity", self.algebraic_connectiveity())

    def average_degree(self):
        return round(self.number_of_edges * 2 / self.number_of_nodes, 4)

    def clustering_coefficient(self):
        return round(nx.average_clustering(self.gra), 4)

    def assortativity_coefficient(self):
        return round(nx.degree_assortativity_coefficient(self.gra), 4)

    def power_law_exponent(self):
        alpha = 0.0
        mind = self.deglist[-1][1]
        for _, deg in self.deglist:
            alpha += math.log(deg / mind)
        return round((1 / alpha) * self.number_of_nodes + 1, 4)

    def edge_distrubution_entropy(self):
        H = 0.0
        m = self.number_of_edges
        for _, deg in self.deglist:
            t = deg / (2 * m)
            H += - t * math.log(t)
        return round(H / math.log(self.number_of_nodes), 4)

    def algebraic_connectiveity(self):
        eigenvector = nx.laplacian_spectrum(self.gra)
        return round(sort(eigenvector)[1], 4)

if __name__ == "__main__":
    dataset_dir = argv[1]
    for dir, dirnames, filenames in os.walk(dataset_dir):
        print(dir)
        for filename in filenames:
            dataset = os.path.join(dir, filename)
            print(filename)
            try:
                GraphEval(dataset).eval()
            except:
                print("error")
            print()
