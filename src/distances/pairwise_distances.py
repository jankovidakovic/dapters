from scipy.spatial.distance import mahalanobis

from src.distances import euclidean_distance, cosine_distance, jaccard_on_cluster_ids, coral, cmd
from src.types import Domain


def pairwise_distances(
        source: Domain,
        target: Domain
):
    centroid_distances = {
        "euclidean": euclidean_distance(source.centroid, target.centroid),
        "mahalanobis": mahalanobis(source.centroid, target.centroid, source.covmat),
        "cosine_distance": cosine_distance(source.centroid, target.centroid),
    }

    domain_divergence_metrics = {
        "jaccard_distance_on_cluster_ids": jaccard_on_cluster_ids(
            source.cluster_ids,
            target.cluster_ids
        ),
        "coral": coral(
            source.representations,
            target.representations
        ),
        "coral_pca": coral(
            source.pca_representations,
            target.pca_representations
        ),
        "cmd_10": cmd(
            source.representations,
            target.representations,
            n_moments=10
        ),
        "cmd_10_pca": cmd(
            source.pca_representations,
            target.pca_representations,
            n_moments=10
        )
    }

    return {
        "centroid_distances": centroid_distances,
        "domain_divergence_metrics": domain_divergence_metrics
    }