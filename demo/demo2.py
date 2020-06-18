#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: seanjiao
# @Email : seansqjiao@outlook.com

import numpy as np
import random
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from partition_sc import partition_graph


def generate_random_3Dgraph(n_nodes, radius, seed=None):
    if seed is not None:
        random.seed(seed)

    # Generate a dict of positions
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}

    # Create random 3D network
    G = nx.random_geometric_graph(n_nodes, radius, pos=pos)
    return G


def plot_3Dgraph(G, sizes, colors, alphas, plot_edge=True, top_node_indices=None, save_name=None):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=sizes[key], edgecolors='k', alpha=alphas[key])

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            if top_node_indices:
                if j[0] in top_node_indices:
                    color = colors[j[0]]
                elif j[1] in top_node_indices:
                    color = colors[j[1]]
                else:
                    color = 'black'
            else:
                color = 'black'

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            if plot_edge:
                ax.plot(x, y, z, c=color, linewidth=0.8, alpha=0.5)

    # Set the initial view
    ax.view_init(30, 0)

    # Hide the axes
    ax.set_axis_off()

    plt.margins(0, 0, 0)
    if save_name:
        plt.savefig(save_name, dpi=300)
        plt.close()
    else:
        plt.show()


# Define a random graph
N_NODES = 100
N_CLUSTER = 5

# Some parameters
edge_th = 0.25  # If the distance between two nodes is smaller than a threshold value,
                # then there is an edge connecting them
random_seed = 3

# Generate the graph
# The existence of the edges between arbitrary node-pairs is determined by their distance.
# This IS a good simulation of epidemiological models.
graph = generate_random_3Dgraph(n_nodes=N_NODES, radius=edge_th, seed=random_seed)

# Visualize the graph
node_colors = ['black'] * N_NODES
node_alphas = [0.5] * N_NODES
node_sizes = [20] * N_NODES
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas, plot_edge=False, save_name='demo2_e1.png')
node_sizes = [10 + 10 * graph.degree(i) ** 2 for i in range(N_NODES)]
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas, save_name='demo2_e2.png')

# Partition the graph
partitions = partition_graph(graph, N_CLUSTER)

# Visualize the partition results
colors = cm.rainbow(np.linspace(0, 1, N_CLUSTER))
node_colors = [colors[c] for c in partitions]
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas, save_name='demo2_e3.png')


# Visualize the coverage of top N nodes of each partition
def viz_topNnodes_settings(G, partitions, N, viz_nb=False):
    top_node_indices = []
    node_colors = ['black'] * N_NODES
    for k in range(N_CLUSTER):
        indices = np.where(partitions == k)[0]
        degrees = np.array([d for (_, d) in G.degree(indices)])
        top_node_indices_ = indices[np.argsort(degrees)[-N:]]
        top_node_indices += list(top_node_indices_)

        for idx in top_node_indices_:
            node_colors[idx] = colors[k]
            node_alphas[idx] = 0.8
            if viz_nb:
                for j in G.neighbors(idx):
                    if j in indices and j not in top_node_indices:
                        node_colors[j] = colors[k]
    return top_node_indices, node_colors, node_alphas


# Visualize the coverage of top 1 nodes of each partition
top_node_indices, node_colors, node_alphas = viz_topNnodes_settings(graph, partitions, 1)
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas, save_name='demo2_e4.png')
top_node_indices, node_colors, node_alphas = viz_topNnodes_settings(graph, partitions, 1, True)
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas,
             top_node_indices=top_node_indices, save_name='demo2_e5.png')

# Visualize the coverage of top 2 nodes of each partition
top_node_indices, node_colors, node_alphas = viz_topNnodes_settings(graph, partitions, 2)
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas, save_name='demo2_e6.png')
top_node_indices, node_colors, node_alphas = viz_topNnodes_settings(graph, partitions, 2, True)
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas,
             top_node_indices=top_node_indices, save_name='demo2_e7.png')

# Visualize the coverage of top 3 nodes of each partition
top_node_indices, node_colors, node_alphas = viz_topNnodes_settings(graph, partitions, 3)
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas, save_name='demo2_e8.png')
top_node_indices, node_colors, node_alphas = viz_topNnodes_settings(graph, partitions, 3, True)
plot_3Dgraph(graph, node_sizes, node_colors, node_alphas,
             top_node_indices=top_node_indices, save_name='demo2_e9.png')

