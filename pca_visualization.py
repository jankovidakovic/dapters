import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src.utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class CLI:
    customer_name: Optional[str]
    config_paths: list[str]
    save_path: str
    message_column: str
    pca_on_first_only: bool


@dataclass
class DatasetConfig:
    raw_dataset_path: str
    processed_dataset_path: str
    cls_representations_path: str


def get_args() -> CLI:
    parser = ArgumentParser("Visualization of hidden representations of transformers.")
    parser.add_argument(
        "--customer_name",
        type=str,
        help="If provided, will be used in the title of the plot.",
    )
    parser.add_argument("--save_path", type=str, default="pca.png")
    parser.add_argument(
        "--message_column",
        type=str,
        default="preprocessed",
        help="Name of the column containing the message.",
    )
    parser.add_argument(
        "--config_paths",
        type=str,
        nargs="+",
        help="Paths to JSON files containing the configs.",
    )
    args = parser.parse_args()
    return CLI(**vars(args))


def main():
    # load model
    args: CLI = get_args()

    setup_logging(args)

    all_representations = []

    configs = [
        DatasetConfig(**json.load(open(config_path, "r")))
        for config_path in args.config_paths
    ]

    all_representations = [
        np.load(config.cls_representations_path) for config in configs
    ]

    # now do the PCA
    pca = PCA(n_components=2)
    if args.pca_on_first_only:
        logger.warning(f"Fitting PCA on first dataset only.")
        pca.fit(all_representations[0])
    else:
        logger.warning(f"Fitting PCA on all data.")
        pca.fit(np.concatenate(all_representations, axis=0))

    pca_representations = [
        pca.transform(embeddings) for embeddings in all_representations
    ]

    centroids = [r.mean(axis=0) for r in pca_representations]

    joint_centroid = np.concatenate(pca_representations, axis=0).mean(axis=0)

    # create color map
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(configs))]

    plt.grid()

    # plot
    for hidden_repr, color, config in zip(pca_representations, colors, configs):
        plt.scatter(
            hidden_repr[:, 0],
            hidden_repr[:, 1],
            marker=".",
            alpha=0.5,
            label=f"{os.path.basename(config.raw_dataset_path).split('.')[0]}",
            color=color,
        )  # could also be abstracted away, but for now who cares

    for color, centroid in zip(colors, centroids):
        # plot center as a cross
        plt.scatter(
            centroid[0],
            centroid[1],
            marker="^",
            edgecolors="black",
            linewidths=1,
            color=color,
            s=100,
        )

    # plot joint centroid
    plt.scatter(
        joint_centroid[0],
        joint_centroid[1],
        marker="x",
        linewidths=3,
        color="black",
        s=100,
    )

    plt.xlabel(
        f"1st principal component ({pca.explained_variance_ratio_[0] * 100:.2f}% $\sigma^2$)"
    )  # noqa
    plt.ylabel(
        f"2nd principal component ({pca.explained_variance_ratio_[1] * 100:.2f}% $\sigma^2$)"
    )
    total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    plt.title(
        f"[CLS] embeddings (PCA). customer={args.customer_name}; N={len(all_representations[0])}. {total_variance * 100:.2f}% $\sigma^2$"
    )

    plt.legend()
    plt.savefig(args.save_path)

    logger.warning(f"Saved plot to {args.save_path}.")


if __name__ == "__main__":
    main()
