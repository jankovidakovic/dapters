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
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoModel, PreTrainedTokenizer, AutoTokenizer, DefaultDataCollator
from transformers.utils import PaddingStrategy

from src.preprocess import to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from src.preprocess.steps import keep, maybe_sample, log_size, deduplication
from src.utils import setup_logging, get_tokenization_fn, pipeline, get_representations, get_cls_token


@dataclass
class CLI:
    perplexity: Optional[int]
    customer_name: Optional[str]
    save_path: str
    message_column: str
    dataset_paths: list[str]
    sample_size: int
    cache_dir: str
    model_name_or_path: str
    tokenizer_path: str
    batch_size: int
    learning_rate: str | float


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
        "--save_path",
        type=str,
        default="tsne.png"
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
    parser.add_argument(
        "--perplexity",
        type=int,
        help="tSNE perplexity"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default="auto"
    )
    args = parser.parse_args()
    return CLI(**vars(args))


logger = logging.getLogger(__name__)


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
    tsne = TSNE(
        n_components=2,
        method="exact",
        perplexity=args.perplexity,
        verbose=2,
        learning_rate=args.learning_rate
    )
    representations = [
        r for r in np.split(
            tsne.fit_transform(np.concatenate(all_embeddings, axis=0)),
            np.cumsum([len(e) for e in all_embeddings]))
    ]  # works like a charm
    if len(representations[-1]) == 0:
        representations = representations[:-1]
    logger.warning(f"tSNE sizes: {pformat([len(r) for r in representations])}")

    # compute cetroids
    centroids = [r.mean(axis=0) for r in representations]

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
    for k, tsne_rep in enumerate(representations):
        plt.scatter(
            tsne_rep[:, 0],
            tsne_rep[:, 1],
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

    plt.xlabel(f"t-SNE 1st dimension")
    plt.ylabel(f"t-SNE 2nd dimension")
    plt.title(f"[CLS] embeddings (t-SNE). customer={args.customer_name}; N={args.sample_size}. perplexity={args.perplexity}")

    plt.legend()
    plt.savefig(args.save_path)

    logger.warning(f"Saved plot to {args.save_path}.")


if __name__ == "__main__":
    main()

