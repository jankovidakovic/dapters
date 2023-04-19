from math import ceil
import numpy as np

def make_same_size(source, target):
    src_batch_size = source.shape[0]
    tgt_batch_size = target.shape[0]
    batch_size = src_batch_size + tgt_batch_size
    src_repeats = ceil(batch_size / src_batch_size)
    tgt_repeats = ceil(batch_size / tgt_batch_size)
    # handle case when source and target are not of same size
    src_embed_rep = np.concatenate([source] * src_repeats, axis=0)[:batch_size]
    tgt_embed_rep = np.concatenate([target] * tgt_repeats, axis=0)[:batch_size]

    return src_embed_rep, tgt_embed_rep


def moment_diff(src, tgt, moment):
    """
    difference between moments  ????????
    """
    ss1 = np.mean(src ** moment, axis=0)
    ss2 = np.mean(tgt ** moment, axis=0)
    return np.linalg.norm(ss1 - ss2)
