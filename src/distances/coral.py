# adopted from https://github.com/CPJKU/da/blob/main/da/coral.py
import numpy as np

from src.distances.utils import make_same_size


def coral(source, target):
    source_rep, target_rep = make_same_size(source, target)
    hidden_dim = source_rep.shape[1]
    src_cov = np.cov(source_rep, rowvar=False)
    tgt_cov = np.cov(target_rep, rowvar=False)

    # squared matrix frobenius norm
    coral_value = np.sum((src_cov - tgt_cov)**2)
    coral_value = coral_value / (4 * hidden_dim ** 2)
    return coral_value
