import numpy as np

from src.distances.utils import moment_diff


def cmd(source, target, n_moments: int):
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)

    source_centered = source - source_centroid
    target_centered = target - target_centroid

    cmd_value = np.linalg.norm(source_centroid - target_centroid)
    cmd_value += sum(moment_diff(source_centered, target_centered, k) for k in range(2, n_moments+1))

    return cmd_value
