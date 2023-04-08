import json
from pprint import pformat
from typing import Callable
import pandas as pd
import logging
import os

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)


def pipeline(*fs) -> Callable:
    """Performs a forward composition of given functions.

    For example, if some three functions are
        A : x -> y
        B : y -> z
        C : z -> w

    then pipeline([A, B, C]) will return a function F : x -> w

    :param fs:  Functions to compose.
    :return:  A callable object that when called, calls the given functions
             in a sequential order and returns the result.
    """

    def pipe(*args, **kwargs):
        output = fs[0](*args, **kwargs)
        for f in fs[1:]:
            output = f(output)
        return output

    return pipe


def sample_by(column: str, sample_size: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def apply(df: pd.DataFrame):
        return df.groupby(column).sample(sample_size)

    return apply


def dummy_preprocess_one() -> Callable[[pd.DataFrame], pd.DataFrame]:
    return pipeline(
        lambda df: df.drop(
            columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags"]
        ),
        lambda df: df.drop_duplicates(subset="message"),
        sample_by("cluster_id", 1),
        lambda df: df.drop(columns=["cluster_id"]),
    )


def make_logfile_name(args):
    return args.log_path


def setup_logging(args):
    # logging
    dirname = os.path.dirname(os.path.abspath(args.log_path))
    os.makedirs(dirname, exist_ok=True)
    log_filename = make_logfile_name(args)

    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # log info only on main process
        level=logging.INFO,  # TODO - info only if verbose?
        handlers=[
            logging.FileHandler(filename=log_filename, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging to file: {os.path.abspath(log_filename)}")


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


def get_labels(labels_path) -> list[str]:
    # load labels
    with open(labels_path, "r") as f:
        labels = json.load(f)

    # sort labels
    labels = sorted(labels["labels"])

    logger.info(f"Sorted labels = {pformat(labels)}")

    return labels


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


def save_checkpoint(
        model: PreTrainedModel,
        output_dir: str,
        global_step: int,
        tokenizer: PreTrainedTokenizer,
):
    output_dir = os.path.join(output_dir, f"checkpoint-{global_step}")  # moze
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Saved model checkpoint to {os.path.abspath(output_dir)}")

    tokenizer.save_pretrained(output_dir)
    logger.warning(f"Saved tokenizer to {os.path.abspath(output_dir)}")
