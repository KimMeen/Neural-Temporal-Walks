import random
import numpy as np
from numba import jit
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


@jit(nopython=True)
def seq_binary_sample(ngh_binomial_prob, num_neighbor):
    sampled_idx = []
    for j in range(num_neighbor):
        idx = seq_binary_sample_one(ngh_binomial_prob)
        sampled_idx.append(idx)
    sampled_idx = np.array(sampled_idx)
    return sampled_idx


@jit(nopython=True)
def seq_binary_sample_one(ngh_binomial_prob):
    seg_len = 10
    a_l_seg = np.random.random((seg_len,))
    seg_idx = 0
    for idx in range(len(ngh_binomial_prob)-1, -1, -1):
        a = a_l_seg[seg_idx]
        seg_idx += 1
        if seg_idx >= seg_len:
            a_l_seg = np.random.random((seg_len,))
            seg_idx = 0
        if a < ngh_binomial_prob[idx]:
            return idx
    return 0


@jit(nopython=True)
def bisect_left_adapt(a, x):
    lo = 0
    hi = len(a)
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo