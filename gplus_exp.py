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


def encode_graph_file():
    ids = []
    edges_dummy = []
    fnames = [fn for fn in os.listdir('gplus') if 'edges' in fn]
    for i, fn in tqdm(enumerate(fnames)):
        # print('{}/{}: {}'.format(i+1, len(fnames), fn))
        with open(osp.join('gplus', fn), 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            series = pd.Series(lines)
            series = series.str.rstrip('\n')
            series = series.str.split()
            edges_ = list(series)

            ids_ = np.array(edges_).reshape([-1])
            ids_ = list(set(list(ids_)))

            ids += ids_
            edges_dummy += edges_
        # break
    ids = list(set(list(ids)))
    n_nodes = len(ids)

    id2num = {}
    for i, id in tqdm(enumerate(ids)):
        id2num[id] = str(i + 1)

    nb = {}
    for e in tqdm(edges_dummy):
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

    with open('metis-5.1.0/gplus.txt', 'w') as f:
        f.write(str(n_nodes) + ' ' + str(n_edges))
        for i in range(n_nodes):
            nb_ = ' '.join(nb[str(i + 1)])
            line = '\n' + nb_
            f.write(line)


def evaluate_partitioning():
    n_parts = 102
    n_top = 3
    graph_file = 'metis-5.1.0/gplus.txt'
    part_file = 'metis-5.1.0/gplus.txt.part.{}'.format(n_parts)
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
