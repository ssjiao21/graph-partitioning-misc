#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: seanjiao
# @Email : seansqjiao@outlook.com

import os
import os.path as osp
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import time


def encode_graph_file():
    with open('as-skitter.txt', 'r') as f:
        lines = f.readlines()
        series = pd.Series(lines)
        series = series.str.rstrip('\n')[5:]
        series = series.str.split()
        edges = list(series)

        ids = np.array(edges).reshape([-1])
        ids = list(set(list(ids)))

    ids = list(set(list(ids)))
    n_nodes = len(ids)

    id2num = {}
    for i, id in tqdm(enumerate(ids)):
        id2num[id] = str(i + 1)
    print(id2num)

    nb = {}
    for e in tqdm(edges):
        nums = [id2num[uid] for uid in e]
        for k in range(2):
            num1 = nums[k]
            num2 = nums[1-k]
            if num1 not in nb:
                nb[num1] = []
            if num2 not in nb[num1]:
                nb[num1].append(num2)

    n_nb = [len(nb[num]) for num in nb]
    n_edges = sum(n_nb) / 2

    print('Number of nodes:', n_nodes)
    print('Number of edges:', n_edges)

    with open('metis-5.1.0/skitter.txt', 'w') as f:
        f.write(str(n_nodes) + ' ' + str(n_edges))
        for i in range(n_nodes):
            nb_ = ' '.join(nb[str(i + 1)])
            line = '\n' + nb_
            f.write(line)


def evaluate_partitioning():
    n_parts = 1696
    n_top = 3
    graph_file = 'metis-5.1.0/skitter.txt'
    part_file = 'metis-5.1.0/skitter.txt.part.{}'.format(n_parts)
    np.set_printoptions(threshold=np.inf)

    with open(graph_file, 'r') as f:
        nbs = f.readlines()[1:]
    nbs = [nb.rstrip('\n').split(' ') for nb in nbs]
    n_nb = [len(nb) for nb in nbs]
    n_nb = np.array(n_nb)

    print('***** Top {} nodes in the whole graph with most neighbors *****'.format(n_parts * n_top))
    cover = []
    idices = list(np.argsort(n_nb)[-n_parts * n_top:])
    for idx in idices:
        cover.append(str(idx + 1))
        cover += nbs[idx]
    cover = list(set(cover))
    print('Number of covered nodes:{}'.format(len(cover)))
    print('{}% coverage'.format(float(len(cover)) * 100 / len(nbs)))

    with open(part_file, 'r') as f:
        parts = f.readlines()
    parts = [int(p.rstrip('\n')) for p in parts]
    parts = np.array(parts)

    print('***** Top {} nodes in each sub-graph ({}) with most neighbors *****'.format(n_top, n_parts))
    cover = []
    for i in tqdm(range(n_parts)):
        # print('Partition' + str(i))
        idices = np.where(parts == i)[0]
        # print('Number of nodes: {}'.format(len(idices)))
        n_nb_ = np.zeros_like(n_nb)
        n_nb_[idices] = n_nb[idices]
        # print(np.max(n_nb_))
        top_n_idx = np.argsort(n_nb_)[-n_top:]
        for idx in top_n_idx:
            cover.append(str(idx + 1))
            cover += nbs[idx]
    cover = list(set(cover))
    print('Number of covered nodes:{}'.format(len(cover)))
    print('{}% coverage'.format(float(len(cover)) * 100 / len(nbs)))


if __name__ == '__main__':
    # encode_graph_file()
    evaluate_partitioning()
