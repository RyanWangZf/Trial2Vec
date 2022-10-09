import pdb

import numpy as np

def precision(ranked_label_list, k):
    return np.mean(ranked_label_list[:,:k].sum(1) / k)

def recall(ranked_label_list, k):
    return (ranked_label_list[:,:k].sum(1) / ranked_label_list.sum(1)).mean()

def ndcg(ranked_label_list, k=None):
    if k is None: k = ranked_label_list.shape[1]
    discount = np.log2(np.arange(2,2+k))
    dcg = np.sum(ranked_label_list[:,:k] / discount, 1)
    idcg = np.sum(np.flip(np.sort(ranked_label_list.copy()), 1)[:,:k] / discount,1)
    ndcg = (dcg / idcg).mean()
    return ndcg