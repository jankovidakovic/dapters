import logging
import os.path
from argparse import ArgumentParser
from pprint import pformat

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DefaultDataCollator

from src.preprocess.steps import multihot_to_list, to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from src.trainer import do_predict
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
        "--eval_dataset_path",
        type=str,
        help="Path to the evaluation dataset.",
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
        "--save_path",
        type=str,
        help="Path to which the dataset will be saved."
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
    )  # moze

    do_tokenize = get_tokenization_fn(
        tokenizer=tokenizer,
        padding="max_length",
        truncation=True,
        max_length=64,
        message_column="preprocessed"
    )  # TODO - unhardcode

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
    )  # ma moze

    model = torch.compile(model).to("cuda")

    logger.warning(f"Model successfully loaded. ")

    df = pd.read_csv(args.eval_dataset_path)

    # load datasets
    do_preprocess = pipeline(
        multihot_to_list(
            label_columns=labels,
            result_column="labels"
        ),
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns)
    )  # looking good

    dataset = do_preprocess(df)

    logger.warning(f"Datasets successfully preprocessed.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=DefaultDataCollator(return_tensors="pt")
    )

    # obtain predictions
    predictions, _ = do_predict(model, dataloader)

    # run predictions through sigmoid to obtain probabilities
    predictions = torch.sigmoid(predictions)

    print(f"{predictions.shape = }")

    df[labels] = predictions

    logger.warning(F"Saving dataframe with predictions to {args.save_path}")

    df.to_csv(args.save_path, index=False)

    logger.warning(F"Dataframe with predictions successfully saved.")


if __name__ == "__main__":
    main()
