import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pprint import pformat
from typing import Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DefaultDataCollator, PreTrainedModel, PreTrainedTokenizer
from transformers.utils import PaddingStrategy

from src.preprocess.steps import keep, deduplication, to_hf_dataset, hf_map, convert_to_torch, \
    sequence_columns, maybe_sample, log_size

from src.utils import pipeline, get_tokenization_fn, setup_logging, get_cls_token, get_representations

logger = logging.getLogger(__name__)


@dataclass
class CLI:
    customer_name: Optional[str]
    save_path: str
    message_column: str
    dataset_paths: list[str]
    sample_size: int
    cache_dir: str
    model_name_or_path: str
    tokenizer_path: str
    batch_size: int
    pca_on_first_only: bool


def get_args() -> CLI:
    parser = ArgumentParser(
        "Visualization of hidden representations of transformers."
    )
    parser.add_argument(
        "--customer_name",
        type=str,
        help="If provided, will be used in the title of the plot."
    )
    parser.add_argument(
        "--pca_on_first_only",
        action="store_true",
        help="If set, PCA will be performed only on the first dataset."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="pca.png"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data2/jvidakovic/.cache/huggingface",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size."
    )
    parser.add_argument(
        "--message_column",
        type=str,
        default="message",
        help="Name of the column containing the message."
    )
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to datasets."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Sample size."
    )
    args = parser.parse_args()
    return CLI(**vars(args))


def main():
    # load model
    args: CLI = get_args()

    setup_logging(args)

    model: PreTrainedModel = AutoModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir
    )

    model = torch.compile(model).to("cuda")  # noqa
    model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        cache_dir=args.cache_dir,
        do_lower_case=True,
    )

    do_tokenize = get_tokenization_fn(
        tokenizer,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=True,
        max_length=64,
        message_column=args.message_column
        # return_special_tokens_mask=True
    )

    # create the dataset processing pipeline
    do_process = pipeline(
        pd.read_csv,
        keep([args.message_column]),
        deduplication(args.message_column),
        maybe_sample(args.sample_size),  # because dataset could be smaller
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns),
        log_size
    )

    all_embeddings = []
    dataset_names = [os.path.basename(p).split(".")[0] for p in args.dataset_paths]

    for k, dataset_path in enumerate(args.dataset_paths):

        dataset_name = dataset_names[k]
        dataset = do_process(dataset_path)
        logger.warning(f"{len(dataset) = }")
        logger.warning(f"{dataset[0] = }")

        logger.warning(f"{dataset_name} finished processing.")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=DefaultDataCollator(return_tensors="pt")
        )

        all_embeddings.append(get_representations(
            model,
            dataloader,
            n_examples=len(dataset),
            semantic_composition=get_cls_token
        ))

    # now do the PCA
    pca = PCA(n_components=2)
    if args.pca_on_first_only:
        logger.warning(f"Fitting PCA on first dataset only.")
        pca.fit(all_embeddings[0])
    else:
        logger.warning(f"Fitting PCA on all data.")
        pca.fit(np.concatenate(all_embeddings, axis=0))

    pca_representations = [
        pca.transform(embeddings) for embeddings in all_embeddings
    ]

    centroids = [r.mean(axis=0) for r in pca_representations]

    joint_centroid = np.concatenate(pca_representations, axis=0).mean(axis=0)

    # compute pairwise distances
    for i, c1 in enumerate(centroids):
        for j, c2 in enumerate(centroids):
            if i == j:
                continue
            logger.warning(f"d({dataset_names[i]},{dataset_names[j]})={np.linalg.norm(c1 - c2)}")

    # create color map
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(args.dataset_paths))]

    plt.grid()

    # plot
    for k, pca_repr in enumerate(pca_representations):
        plt.scatter(
            pca_repr[:, 0],
            pca_repr[:, 1],
            marker=".",
            alpha=0.5,
            label=f"{os.path.basename(args.dataset_paths[k]).split('.')[0]}",
            color=colors[k]
        )  # could also be abstracted away, but for now who cares

    for k, centroid in enumerate(centroids):
        # plot center as a cross
        plt.scatter(
            centroid[0],
            centroid[1],
            marker="^",
            edgecolors="black",
            linewidths=1,
            color=colors[k],
            s=100
        )

    # plot joint centroid
    plt.scatter(
        joint_centroid[0],
        joint_centroid[1],
        marker="^",
        edgecolors="black",
        linewidths=1,
        color="black",
        s=100
    )

    plt.xlabel(f"1st principal component ({pca.explained_variance_ratio_[0] * 100:.2f}% $\sigma^2$)")   # noqa
    plt.ylabel(f"2nd principal component ({pca.explained_variance_ratio_[1] * 100:.2f}% $\sigma^2$)")
    total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    plt.title(f"[CLS] embeddings (PCA). customer={args.customer_name}; N={args.sample_size}. {total_variance * 100:.2f}% $\sigma^2$")

    plt.legend()
    plt.savefig(args.save_path)

    logger.warning(f"Saved plot to {args.save_path}.")


if __name__ == "__main__":
    main()