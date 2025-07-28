import math
import numpy as np

# this is for only one ground truth.
def ndcg(ranks, label, k):
    if label in ranks[:k]:
        label_rank = ranks.index(label)
        return 1.0/math.log2(label_rank + 2)
    else:
        return 0

def recall(ranks, label, k):
    if label in ranks[:k]:
        return 1
    else:
        return 0

def mrr(ranks, label, k):
    if label in ranks[:k]:
        label_rank = ranks.index(label)
        return 1.0/(label_rank+1)
    else:
        return 0
