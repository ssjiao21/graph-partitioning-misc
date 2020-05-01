#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: seanjiao
# @Email : seansqjiao@outlook.com

import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import SpectralClustering


N_NODES = 30
N_CLUSTER = 5

G = nx.gnp_random_graph(N_NODES, 0.1, seed=5)

edges = G.edges()
# print(edges)

A = np.zeros((N_NODES, N_NODES))
for e in edges:
    A[e[0], e[1]] = 1
    A[e[1], e[0]] = 1

clustering = SpectralClustering(
    n_clusters=N_CLUSTER, assign_labels="discretize", affinity="precomputed", random_state=0)
ts = time.time()
clustering.fit(A)
print('Time elapsed: {}s'.format(time.time() - ts))
# print(clustering.labels_)

colors = cm.rainbow(np.linspace(0, 1, N_CLUSTER))
node_colors = [colors[c] for c in clustering.labels_]

nx.draw_networkx(G, node_size=5, node_color=node_colors, with_labels=False)
# plt.show()
plt.savefig("undirected_graph.png", dpi=1000)
