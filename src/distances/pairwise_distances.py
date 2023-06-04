import logging
from typing import Optional

import numpy as np
from scipy.spatial.distance import mahalanobis

from src.distances import euclidean_distance, cosine_distance, jaccard_on_cluster_ids, coral, cmd
from src.types import DomainCollection


logger = logging.getLogger(__name__)


def compute_centroid_distances(
        domain_collection: DomainCollection,
        source: int,
        target: int
):
    return {
        "euclidean": euclidean_distance(domain_collection[source].centroid, domain_collection[target].centroid),
        "mahalanobis": mahalanobis(
            domain_collection[source].centroid,
            domain_collection[target].centroid,
            np.linalg.inv(domain_collection.representation_covmat)),
        "cosine_distance": cosine_distance(
            domain_collection[source].centroid,
            domain_collection[target].centroid),
    }


def compute_domain_divergences(
        domain_collection: DomainCollection,
        source: int,
        target: int,
):
    return {
        "coral": coral(
            domain_collection[source].representations,
            domain_collection[target].representations
        ),
        "cmd_10": cmd(
            domain_collection[source].representations,
            domain_collection[target].representations,
            n_moments=10
        ),
        "cmd_100": cmd(
            domain_collection[source].representations,
            domain_collection[target].representations,
            n_moments=100
        )
    }



def compute_pairwise_distances(
        domain_collection: DomainCollection,
        source: int = 0,
        target: int = 1
):
    centroid_distances = compute_centroid_distances(domain_collection, source, target)

    domain_divergence_metrics = compute_domain_divergences(domain_collection, source, target)

    return {
        "centroid_distances": centroid_distances,
        "domain_divergence_metrics": domain_divergence_metrics
    }