import logging
from argparse import ArgumentParser
import json

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
        pca_dim=50
    )

    logger.warning(f"Domain collection successfully initialized.")

    # compute dummy pairwise distances
    distances = compute_pairwise_distances(
        domain_collection,
        source=0,
        target=1
    )

    print(json.dumps(distances, indent=2))


if __name__ == '__main__':
    main()
