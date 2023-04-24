import itertools
import logging
import os.path
from argparse import ArgumentParser
import json

from scipy.spatial.distance import mahalanobis

from src.distances import euclidean_distance, cosine_distance
from src.distances.pairwise_distances import compute_pairwise_distances
from src.types import DomainCollection, HiddenRepresentationConfig
from src.utils import setup_logging, get_domain_from_config

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser("Computation of pairwise distances")
    parser.add_argument(
        "--config_paths",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--pca_n_components",
        type=int,
        default=50,
        help="Number of components to use for PCA. Defaults to 50"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to which metrics will be saved."
    )
    parser.add_argument(
        "--redacted",
        action="store_true",
        help="If set, will censor out any NDA-sensitive information from the plots."
    )

    args = parser.parse_args()
    setup_logging(args)

    # load configs
    configs = (json.load(open(path, "r")) for path in args.config_paths)
    configs = [HiddenRepresentationConfig(**config) for config in configs]
    logger.info(f"Loaded {len(configs)} configs.")

    domains = [
        get_domain_from_config(config)
        for config in configs
    ]

    domain_collection = DomainCollection(
        domains,
        pca_dim=args.pca_n_components
    )

    logger.warning(f"Domain collection successfully initialized.")

    # compute dummy pairwise distances
    distances = {}

    for source in range(len(domain_collection)):
        for target in range(len(domain_collection)):
            if source == target:
                continue
            if args.redacted:
                key = f"({source}, {target})"
            else:
                key = f"({domain_collection[source].name}, {domain_collection[target].name})"
            distances[key] = compute_pairwise_distances(
                domain_collection,
                source=source,
                target=target
            )

    distances_to_joint_centroid = {}

    for i, domain in enumerate(domain_collection):
        if args.redacted:
            key = f"Dataset {i}"
        else:
            key = domain.name
        distances_to_joint_centroid[key] = {
            "euclidean": euclidean_distance(domain.centroid, domain_collection.joint_centroid),
            "mahalanobis": mahalanobis(
                domain.centroid,
                domain_collection.joint_centroid,
                domain_collection.representation_covmat),
            "cosine_distance": cosine_distance(
                domain.centroid,
                domain_collection.joint_centroid
            ),
        }

    distances["distances_to_joint_centroid"] = distances_to_joint_centroid

    print(json.dumps(distances, indent=2))

    with open(args.save_path, "w") as f:
        json.dump(distances, f, indent=2)

    logger.warning(f"Distances successfully saved to {os.path.abspath(args.save_path)}")


if __name__ == '__main__':
    main()
