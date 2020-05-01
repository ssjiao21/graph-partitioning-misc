#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: seanjiao
# @Email : seansqjiao@outlook.com


import pandas as pd
import numpy as np

data = pd.read_csv('gps_20161130.csv')
df = pd.DataFrame(data)

print(df.shape)
print(df['Timeframe'].values.dtype)

time = 20161129096
df = df.iloc[np.where(df['Timeframe'] == time)]
# print(df.shape)
df = df[['CarId', 'Lng', 'Lat']]
# print(df[:10])


def lonlat2wgt(point1, point2):
    """
    Weighting scheme: Gaussian kernel
    :param point1: (lon, lat)
    :param point2: (lon, lat)
    :return:
    """
    dist = 111.32 * np.sqrt(np.sum((point1 - point2) ** 2))
    wgt = 1 / dist ** 2
    return wgt


n_nodes = df.values.shape[0]
for i in range(n_nodes):
    point1 = df.iloc[i, 1:].values
    nodes = []
    for j in range(n_nodes):
        if j==i:
            continue
        point2 = df.iloc[j, 1:].values
        wgt = lonlat2wgt(point1, point2)

        node = str(j + 1) + ' ' + str(wgt)
        nodes.append(node)
    line = ' '.join(nodes)
    print(line)
    break

n_edges = n_nodes * (n_nodes)

with open('metis-5.1.0/car.txt', 'w') as f:
    f.write(str(n_nodes) + ' ' + str(n_edges))
    for i in range(n_nodes):
        nb_ = ' '.join(nb[str(i + 1)])
        line = '\n' + nb_
        f.write(line)