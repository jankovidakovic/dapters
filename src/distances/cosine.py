import numpy as np


def cosine_similarity(source, target):
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    return np.dot(source, target) / (source_norm * target_norm)


def cosine_distance(source, target):
    return 1 - cosine_similarity(source, target)
