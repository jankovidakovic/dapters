import logging
from argparse import ArgumentParser
from pprint import pformat

import mlflow
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DefaultDataCollator

from src.preprocess.steps import multihot_to_list, to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from src.trainer import evaluate_finetuning
from src.utils import get_labels, pipeline, get_tokenization_fn, setup_logging


logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser("Fine-tuning evaluation")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to the tokenizer",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--source_domain_dataset_path",
        type=str,
        help="Path to the source domain dataset.",
    )
    parser.add_argument(
        "--target_domain_dataset_path",
        type=str,
        help="Path to the target domain dataset.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        help="Path to JSON file containing the labels."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation. Defaults to 128."
    )
    parser.add_argument(
        "--mlflow_run_id",
        type=str,
        help="MLFlow run ID to log metrics to."
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:34567",
        help="MLFlow tracking URI."
    )

    return parser


def main():
    args = get_parser().parse_args()
    setup_logging(None)

    logger.info(f"Setting up the tokenizer...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=64,
        do_lower_case=True,
    )

    do_tokenize = get_tokenization_fn(
        tokenizer=tokenizer,
        padding="max_length",
        truncation=True,
        max_length=64,
        message_column="preprocessed"
    )

    logger.info(f"Tokenizer set up.")

    # load labels
    labels = get_labels(args.labels_path)

    logger.warning(f"{labels = }")
    logger.warning(f"{len(labels) = }")

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        problem_type="multi_label_classification",
        num_labels=len(labels)
    )

    model = torch.compile(model).to("cuda")

    logger.warning(f"Model successfully loaded. ")

    # load datasets
    do_preprocess = pipeline(
        pd.read_csv,
        DataFrame.dropna,
        multihot_to_list(
            label_columns=labels,
            result_column="labels"
        ),
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns)
    )

    source_dataset = do_preprocess(args.source_domain_dataset_path)
    target_dataset = do_preprocess(args.target_domain_dataset_path)

    logger.warning(f"Datasets successfully preprocessed.")

    source_dataloader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=DefaultDataCollator(return_tensors="pt")
    )

    target_dataloader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=DefaultDataCollator(return_tensors="pt")
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.start_run(
        run_id=args.mlflow_run_id,
    )

    do_evaluate = evaluate_finetuning()

    logger.warning(f"Evaluating on source domain ({args.source_domain_dataset_path})...")

    source_metrics = do_evaluate(
        model=model,
        eval_dataloader=source_dataloader,
        metrics_prefix="source"
    )

    logger.warning(f"Source evaluation finished.")
    logger.warning(pformat(source_metrics))

    logger.warning(f"Evaluating on target domain ({args.target_domain_dataset_path})...")

    target_metrics = do_evaluate(
        model=model,
        eval_dataloader=target_dataloader,
        metrics_prefix="target"
    )

    logger.warning(f"Target evaluation finished.")
    logger.warning(pformat(target_metrics))

    mlflow.log_metrics(source_metrics)
    mlflow.log_metrics(target_metrics)

    logger.warning(f"Evaluation finished.")


if __name__ == "__main__":
    main()