# input: path to (multiple) datasets
# output: dataframe with hidden_representations
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional
import os

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DefaultDataCollator,
)
import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers.utils import PaddingStrategy
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
    parser.add_argument("--sample_size", type=int, default=1000, help="Sample size.")
    # TODO - add semantic_composition as an argument
    args = parser.parse_args()
    return CLI(**vars(args))


def main():
    args: CLI = get_args()
    setup_logging(None)

    model: PreTrainedModel = AutoModel.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )

    model = torch.compile(model).to("cuda")
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
        maybe_sample(args.sample_size),
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

        logger.warning(f"Pandas processing started.")
        df_final = process_pd(dataset_path)
        logger.warning("Pandas processing finished.")

        logger.warning(f"Huggingface processing started.")
        dataset = process_hf(df_final)
        logger.warning("Huggingface processing finished.")

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

        logger.warning(f"Embeddings computed for {dataset_name}.")
        df_final.loc[:, "embeddings"] = embeddings
        save_path = os.path.join(args.save_dir, f"{dataset_name}.csv")
        df_final.to_csv(save_path)
        logger.info(f"Dataset {dataset_name} successfully saved to {save_path}")

    logger.warning(f"Inference finished.")


if __name__ == "__main__":
    main()
