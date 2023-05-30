import logging
import torch
import os.path
from argparse import ArgumentParser
from pprint import pformat

import mlflow
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DefaultDataCollator

from src.distances.pairwise_distances import compute_pairwise_distances
from src.preprocess.steps import multihot_to_list, to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from src.trainer import evaluate_finetuning, do_predict, compute_metrics
from src.types import Domain, DomainCollection
from src.utils import setup_logging, get_labels, get_tokenization_fn, pipeline

logger = logging.getLogger(__name__)


def main():
    # TODO - we need to be able to load the adapter setup as well
    #   TODO -> just reuse hydra config

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    parser = ArgumentParser("Evaluation of saved checkpoint on multiple domains")

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Paths to the checkpoints to evaluate.",
    )

    parser.add_argument(
        "--source_dataset_path",
        help="Filesystem path to the evaluation dataset from source domain"
    )

    parser.add_argument(
        "--target_dataset_path",
        help="Filesystem path to the evaluation dataset from target domain"
    )

    parser.add_argument(
        "--source_dataset_name",
        help="Source dataset name to be used in metric logging"
    )

    parser.add_argument(
        "--target_dataset_name",
        help="Target dataset name to be used in metric logging"
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

    parser.add_argument(
        "--labels_path",
        type=str,
        default="./labels.json",
        help="Path to JSON file containing the labels."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation. Default is 128."
    )

    args = parser.parse_args()

    setup_logging(None)

    # load labels
    labels = get_labels(args.labels_path)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.start_run(
        run_id=args.mlflow_run_id,
    )

    checkpoint_name = os.path.abspath(args.checkpoint)
    logger.warning(f"Running evaluation for checkpoint: {checkpoint_name}")

    # extract checkpoint step
    checkpoint_step = args.checkpoint.split("/")[-1].split("-")[0]

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
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

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        problem_type="multi_label_classification",
        num_labels=len(labels)
    )  # sumnjivo tho  -- ma moze

    model = model.to("cuda")  # TODO - device

    do_preprocess = pipeline(
        pd.read_csv,
        multihot_to_list(
            label_columns=labels,
            result_column="labels"
        ),
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns)
    )

    source_dataset = do_preprocess(args.dataset_path[0])
    target_dataset = do_preprocess(args.dataset_paths[1])

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

    source_predictions, source_references, source_hidden_states = do_predict(model, source_dataloader, return_hidden_states=True)
    target_predictions, target_references, target_hidden_states = do_predict(model, target_dataloader, return_hidden_states=True)

    source_metrics = compute_metrics(source_predictions, source_references, "source")
    logger.warning(pformat(source_metrics))
    mlflow.log_metrics(
        source_metrics,
        step=int(checkpoint_step)
    )

    target_metrics = compute_metrics(target_predictions, target_references, "target")
    logger.warning(pformat(target_metrics))
    mlflow.log_metrics(
        target_metrics,
        step=int(checkpoint_step)
    )

    # now we obtain domain shift!

    source_domain = Domain(name="source", representations=source_hidden_states.numpy())
    target_domain = Domain(name="target", representations=target_hidden_states.numpy())

    domain_collection = DomainCollection(domains=[source_domain, target_domain], pca_dim=0.95)

    domain_distances = compute_pairwise_distances(domain_collection)
    logger.warning(pformat(domain_distances))

    mlflow.log_metrics(
        domain_distances["centroid_distances"],
        step=int(checkpoint_step)
    )
    mlflow.log_metrics(
        domain_distances["domain_divergence_metrics"],
        step=int(checkpoint_step)
    )

    # this should actually work now, no?

    mlflow.end_run()


if __name__ == '__main__':
    main()
