# python version 3.8.5
# -*- coding: utf-8 -*-
import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

def iid_sampling(y_train, num_users):
    n_train = y_train.shape[0]
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    for i in range(num_users):
        user_idxs = set(np.random.choice(all_idxs, num_items, replace=False)) # 'replace=False' make sure that there is no repeat
        dict_users[i] = list(user_idxs)
        all_idxs = list(set(all_idxs)-user_idxs)
    return dict_users

def non_iid_dirichlet_sampling(y_train, num_classes, num_users, dirichlet_beta=1.0):
    n_train = y_train.shape[0]
    min_size = 1
    min_require_size = 10
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = np.where(y_train == k)[0] # get the corresponding sample index if label == k
            np.random.shuffle(idx_k) # Shuffle sample index
            proportions = np.random.dirichlet(np.repeat(dirichlet_beta, num_users))
            proportions = np.array([p * (len(idx_j) < n_train / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map
