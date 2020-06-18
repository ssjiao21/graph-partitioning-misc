#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: seanjiao
# @Email : seansqjiao@outlook.com


import numpy as np
import time
import networkx as nx
from sklearn.cluster import SpectralClustering


"""
For simplicity, we use spectral clustering algorithm to partition a demonstration graph. 
This is ONLY for demonstration purpose.

When targeting on real-world datasets, we adopt the Metis tools for partitioning the graphs.
"""


def partition_graph(G, n_cluster):
    edges = G.edges()
    # print(edges)
    n_nodes = G.number_of_nodes()

    A = np.zeros((n_nodes, n_nodes))
    for e in edges:
        A[e[0], e[1]] = 1
        A[e[1], e[0]] = 1

    clustering = SpectralClustering(
        n_clusters=n_cluster, assign_labels="discretize", affinity="precomputed", random_state=0)
    ts = time.time()
    clustering.fit(A)
    print('Time elapsed: {}s'.format(time.time() - ts))
    # print(clustering.labels_)

    return clustering.labels_
