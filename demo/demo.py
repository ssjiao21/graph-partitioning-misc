#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: seanjiao
# @Email : seansqjiao@outlook.com

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from partition_sc import partition_graph

# Define a random graph
N_NODES = 50
N_CLUSTER = 5

# Some parameters
edge_prob = 0.08
random_seed = 3

# Generate the graph
# The existence of the edges between arbitrary node-pairs is fully random.
# This is not a very good simulation of epidemiological models.
graph = nx.gnp_random_graph(N_NODES, edge_prob, seed=random_seed)

# Partition the graph
partitions = partition_graph(graph, N_CLUSTER)

# Visualize the partition results
colors = cm.rainbow(np.linspace(0, 1, N_CLUSTER))
node_colors = [colors[c] for c in partitions]
nx.draw_networkx(graph, node_size=5, node_color=node_colors, with_labels=False)
plt.show()
# plt.savefig("demo.png", dpi=1000)

