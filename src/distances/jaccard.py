import numpy as np


def jaccard_on_cluster_ids(source_cluster_ids, target_cluster_ids):
    intersection_size = len(np.intersect1d(source_cluster_ids, target_cluster_ids))
    return float(intersection_size) / float(len(source_cluster_ids) + len(target_cluster_ids) - intersection_size)
