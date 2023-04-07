import json
import logging
from pprint import pformat

import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

from src.utils import dummy_preprocess_one, get_label_converter

logger = logging.getLogger(__name__)


def setup_data(args, tokenizing_fn, labels: list[str]) -> (Dataset, Dataset):
    """ Sets up the dataset for multilabel classification.

    :param args:  command line args.
    :param tokenizing_fn:  tokenization function
    :param labels: labels
    :return: pytorch dataset
    """

    def setup_split(
            dataframe_path: str
    ) -> Dataset:
        df = pd.read_csv(args.dataframe_path)
        do_preprocess = dummy_preprocess_one()
        df = do_preprocess(df)

        # initialize MLFlow run
        #   -> nah fuck that, I need to finish evaluation first

        # I dont want to use HF trainer because it should be slower than pure pytorch, right?
        #   probably

        # but I can also empirically measure it
        # and also, few things are problematic:
        #   1. early stopping
        #   2. LR scheduling?

        logging.info(pformat(df.head()))

        # convert to HF dataset
        dataset = Dataset.from_pandas(df, preserve_index=False)

        dataset = dataset.map(tokenizing_fn, batched=True)

        # up until here is only the dataset processing -> that can be extracted

        # this can all be abstracted away into some function probably
        #   smth like "setup_context"

        label_converter = get_label_converter(labels)
        dataset = dataset.map(label_converter)

        dataset = dataset.with_format(
            "torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
            # TODO - columns are model-dependent
        )

        logger.info(f"dataset contains {len(dataset)} examples.")
        logger.info(f"dataset example: {pformat(dataset[0])}")

        return dataset

    train_dataset = setup_split(
        args.train_dataset_path
    )

    eval_dataset = setup_split(
        args.eval_dataset_path
    )

    return train_dataset, eval_dataset


def setup_dataloaders(
        train_dataset: Dataset,
        eval_dataset: Dataset,
        args
) -> (DataLoader, DataLoader):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        shuffle=True,  # TODO
        collate_fn=DefaultDataCollator(return_tensors="pt")  # TODO
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=DefaultDataCollator(return_tensors="pt")  # TODO
    )

    return train_dataloader, eval_dataloader
