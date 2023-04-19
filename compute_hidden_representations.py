# input: path to (multiple) datasets
# output: dataframe with hidden_representations
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from typing import Optional
import os
import json

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DefaultDataCollator, set_seed,
)
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers.utils import PaddingStrategy

from src.types import HiddenRepresentationConfig
from src.utils import (
    get_tokenization_fn,
    pipeline,
    setup_logging,
    get_cls_token,
    get_representations,
)

from src.preprocess.steps import (
    keep,
    deduplication,
    to_hf_dataset,
    hf_map,
    convert_to_torch,
    sequence_columns,
    maybe_sample,
    log_size,
)

import logging


logger = logging.getLogger(__name__)


@dataclass
class CLI:
    save_dir: str
    message_column: str
    dataset_paths: list[str]
    sample_size: int
    cache_dir: str
    model_name_or_path: str
    tokenizer_path: str
    batch_size: int
    n_samples: int
    random_seed: int


def get_args() -> CLI:
    parser = ArgumentParser(
        "Script for computing hidden representations of transformers."
    )
    parser.add_argument(
        "--save_dir", type=str, default="./data/processed/cls_representations"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data2/jvidakovic/.cache/huggingface",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--message_column",
        type=str,
        default="message",
        help="Name of the column containing the message.",
    )
    parser.add_argument(
        "--dataset_paths", type=str, nargs="+", required=True, help="Paths to datasets."
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
        help="Sample size of each sample."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of times that the sampling will occur."
             "All samples will be saved."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=19041054,
        help="Random seed used for sampling."
    )
    # TODO - add semantic_composition as an argument
    args = parser.parse_args()
    return CLI(**vars(args))


def main():
    args: CLI = get_args()
    setup_logging(None)
    set_seed(args.random_seed)

    logger.warning(f"Creating save dir ({args.save_dir}) if it doesnt exist...")
    os.makedirs(args.save_dir, exist_ok=True)

    model: PreTrainedModel = AutoModel.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )

    model = torch.compile(model)  # noqa
    model = model.to("cuda")
    model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, cache_dir=args.cache_dir, do_lower_case=True
    )

    do_tokenize = get_tokenization_fn(
        tokenizer,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=True,
        max_length=64,
        message_column=args.message_column,
    )

    process_pd = pipeline(
        pd.read_csv,
        keep([args.message_column, "cluster_id"]),
        deduplication(args.message_column),
        log_size,
    )

    process_hf = pipeline(
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns),
    )

    # dataset_names = [os.path.basename(p).split(".")[0] for p in args.dataset_paths]

    for dataset_path in args.dataset_paths:
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        logger.warning(f"Processing dataset: {dataset_name}")

        hss = []
        dfs = []

        df = process_pd(dataset_path)

        for sample_id in range(args.n_samples):
            logger.warning(f"Sample {sample_id}")

            # sample using a different random seed
            sampler = maybe_sample(args.sample_size, args.random_seed + sample_id)
            df = sampler(df)
            dfs.append(df)

            dataset = process_hf(df)
            logger.warning(f"Finished preprocessing for sample {sample_id}")

            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=DefaultDataCollator(return_tensors="pt"),
            )
            embeddings = get_representations(
                model,
                dataloader,
                n_examples=len(dataset),
                semantic_composition=get_cls_token,
            )
            hss.append(embeddings)

        logger.warning(f"All samples obtained for {dataset_name}.")

        save_path = os.path.abspath(args.save_dir)
        dataset_save_paths = [
            os.path.join(save_path, f"{dataset_name}_{i}")
            for i in range(len(hss))
        ]
        # save_path = os.path.abspath(save_path)
        for d_save_path, hs, df in zip(dataset_save_paths, hss, dfs):
            # save dataframe
            df.to_csv(f"{d_save_path}.csv", index=False)
            # save hidden representations
            np.save(d_save_path, hs)
            logger.warning(f"{d_save_path} successfully saved.")

        # save metadata JSON
        config = HiddenRepresentationConfig(
            name=dataset_name,
            source_dataset=os.path.abspath(dataset_path),
            processed_datasets=[
                f"{save_path}.csv" for save_path in dataset_save_paths
            ],
            cls_representations=[
                f"{save_path}.npy" for save_path in dataset_save_paths
            ]
        )
        json_save_path = os.path.join(args.save_dir, f"info.{dataset_name}.json")
        json_save_path = os.path.abspath(json_save_path)
        with open(json_save_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        logger.warning(f"Metadata JSON file successfully saved to '{json_save_path}'")

    logger.warning(f"Inference finished.")


if __name__ == "__main__":
    main()
