#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: seanjiao
# @Email : seansqjiao@outlook.com

from tqdm import tqdm
import json
import numpy as np
import time
from sklearn.cluster import SpectralClustering

friends = {}
num_of_fri = {}

with open('data.csv', 'r') as f:
    for line in tqdm(f.readlines()[1:]):
        id = line.split(',')[0].lstrip('"').rstrip('"')
        fri_ = line.split(',[')[-1]
        fri_ = fri_.rstrip('" ]').lstrip(' "')
        fri = fri_.split('", "')
        friends[id] = fri
        num_of_fri[id] = len(fri)
# with open('friends.json', 'w') as f:
#     json.dump(friends, f)

ids = list(friends.keys())
N = len(ids)
print('Total:', N)
# print(num_of_fri)

# indexing twitter IDs
id2idx = {}
for i, id in tqdm(enumerate(friends)):
    id2idx[id] = i
# print(id2idx)

N_SAMPLE = 100
A = np.zeros((N_SAMPLE, N_SAMPLE))
for k in tqdm(range(N_SAMPLE)):
    id = ids[k]
    i = id2idx[id]
    for fri_id in friends[id]:
        if fri_id not in id2idx.keys():
            continue
        j = id2idx[fri_id]
        if j >= N_SAMPLE:
            continue
        A[i, j] = 1
        A[j, i] = 1
# D = np.diag(A.sum(axis=1))
# L = D-A
#
# # np.save('Laplacian.npy', L)
#
# # L = np.load('Laplacian.npy')
# D_ = np.max(D, axis=1)
# print(D_[:100])
# print(np.mean(D_))
#
#
# ts = time.time()
# vals, vecs = np.linalg.eig(L)
# print('Time elapsed: {}s'.format(time.time() - ts))
# vecs = vecs[:, np.argsort(vals)]
# vals = vals[np.argsort(vals)]
#
# N_C_ = np.where(np.absolute(vals) > 0.01)
# N_C = len(list(N_C_[0]))
# print(np.absolute(vals)[-10:])
# print('Number of clusters:', N_C)

clustering = SpectralClustering(n_clusters=10, assign_labels="discretize",affinity="precomputed", random_state=0)
ts = time.time()
clustering.fit(A)
print('Time elapsed: {}s'.format(time.time() - ts))
print(clustering.labels_[:100])
