import argparse
import json
import logging
from pprint import pformat

import pandas as pd
import torch
from datasets import Dataset
from torch import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, BertForSequenceClassification, DefaultDataCollator
from transformers.utils import PaddingStrategy

from src.utils import setup_logging, dummy_preprocess_one

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Fine-tuning arguments parser")

parser.add_argument(
    "--dataframe_path", type=str, help="Filesystem path to a dataframe."
)

parser.add_argument(
    "--log_path",
    type=str,
    help="Filesystem path to a log file in which logs will be written.",
)

parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    help="Fully qualified model name, either on Huggingface Model Hub or "
         "a local filesystem path."
)

parser.add_argument(
    "--padding",
    type=str,
    choices=["longest", "max_length", "do_not_pad"],
    default="max_length",
    help="Padding strategy when tokenizing. Defaults to 'max_length'."
)

parser.add_argument(
    "--max_length",
    type=int,
    default=64,
    help="Model max length. Defaults to 64."
)

parser.add_argument(
    "--labels_path",
    type=str,
    help="Path to a JSON file with the layout {'labels': ['LABEL_0', 'LABEL_1', ...]}"
)

parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=4,
    help="Batch size per device. Defaults to 4."
)

parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=4,
    help="Number of dataloader workers. Defaults to 4."
)

parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    help="Number of epochs. Defaults to 1."
)

def get_tokenization_fn(
        tokenizer: PreTrainedTokenizer,
        padding: PaddingStrategy = PaddingStrategy.LONGEST,
        truncation: bool = True,
        max_length: int = 64
):
    def tokenize(examples):
        return tokenizer(
            examples["message"],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt"
        )

    return tokenize


def get_label_converter(labels: list[str]):
    """ Factory for multihot label encoder.

    Given a list of labels, converts multiple columns with label names
    to a single column named 'labels'. The newly created column
    contains multihot encoding of labels.

    :param labels: list of labels
    :return: multihot label encoder
    """

    def get_multihot(example):
        """ Converts multiple label columns into a single multihot encoding.

        :param example: example which contains labels as separate column.
        :return: multihot encoding of all labels, stored in new 'labels' column.
        """

        multihot = torch.zeros(len(labels), dtype=torch.float32)
        for i, label in enumerate(labels):
            if example[label] == 1:
                multihot[i] = 1
        return {
            "labels": multihot
        }

    # batched is a bitch here, lets just do unbatched who cares

    return get_multihot


def main():
    args = parser.parse_args()
    setup_logging(args)

    logging.info(f"Dataset path = {args.dataframe_path}")

    df = pd.read_csv(args.dataframe_path)
    do_preprocess = dummy_preprocess_one()
    df = do_preprocess(df)

    # I dont want to use HF trainer because it should be slower than pure pytorch, right?
    #   probably

    # but I can also empirically measure it
    # and also, few things are problematic:
    #   1. early stopping
    #   2. LR scheduling?

    logging.info(pformat(df.head()))

    # convert to HF dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path
    )

    tokenization_fn = get_tokenization_fn(
        tokenizer=tokenizer,
        padding=args.padding,
        truncation=True,
        max_length=args.max_length
    )

    dataset = dataset.map(tokenization_fn, batched=True)

    # load labels
    with open(args.labels_path, "r") as f:
        labels = json.load(f)

    # sort labels
    labels = sorted(labels["labels"])

    logger.info(f"Sorted labels = {pformat(labels)}")

    label_converter = get_label_converter(labels)
    dataset = dataset.map(label_converter)

    dataset = dataset.with_format(
        "torch",
        columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    )

    # TODO - IterableDataset = ?

    logger.info(f"dataset contains {len(dataset)} examples.")
    logger.info(f"dataset example: {pformat(dataset[0])}")

    # initialize model
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=len(labels),
        problem_type="multi-label"
    )
    model = torch.compile(model)

    logger.info(f"Model loaded successfully.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=DefaultDataCollator(return_tensors="pt")
    )

    optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}") :
            output = model(**batch)
            loss = binary_cross_entropy_with_logits(
                input=output["logits"],
                target=batch["labels"]
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % 10 == 0:
                print(f"loss = {loss}")


if __name__ == "__main__":
    main()
