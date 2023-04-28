import logging
from pprint import pformat

import pandas as pd
from datasets import Dataset, Sequence
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, DataCollator

from src.cli.finetuning import FineTuningArguments
from src.utils import dynamic_import, get_label_converter

logger = logging.getLogger(__name__)


def setup_data(args: FineTuningArguments, tokenizing_fn, labels: list[str]) -> (Dataset, Dataset):
    """ Sets up the dataset for multilabel classification.

    :param args:  command line args.
    :param tokenizing_fn:  tokenization function
    :param labels: labels
    :return: pytorch dataset
    """

    train_dataset = setup_split(tokenizing_fn, labels, args.train_dataset_path, args.preprocessing)

    eval_dataset = setup_split(tokenizing_fn, labels, args.eval_dataset_path, args.preprocessing)

    return train_dataset, eval_dataset


def setup_split(tokenizing_fn, labels, dataframe_path, preprocessing_method: str) -> Dataset:
    df = pd.read_csv(dataframe_path)
    # do_preprocess = dynamic_import()
    preprocessing = dynamic_import("src.preprocess", preprocessing_method)
    logger.warning(f"Using preprocessing method: {preprocessing_method}")
    df = preprocessing(df)

    logging.info(pformat(df.head()))

    # convert to HF dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)

    dataset = dataset.map(tokenizing_fn, batched=True)

    # up until here is only the dataset processing -> that can be extracted

    # this can all be abstracted away into some function probably
    #   smth like "setup_context"

    label_converter = get_label_converter(labels)
    dataset = dataset.map(label_converter)  # I didnt make this batched, fuck it

    # keep only columns which are sequences, discard other columns
    # this way, keeping columns is model-independent (e.g. some models dont have "token_type_ids")
    columns_to_keep = list(map(
        lambda entry: entry[0],  # feature key (name)
        filter(
            lambda entry: type(entry[1]) == Sequence,  # value is a Sequence
            dataset.features.items())))

    dataset = dataset.with_format(
        "torch",
        columns=columns_to_keep
        # TODO - columns are model-dependent
    )

    logger.info(f"dataset contains {len(dataset)} examples.")
    logger.info(f"dataset example: {pformat(dataset[0])}")

    return dataset


def setup_dataloaders(
        train_dataset: Dataset,
        eval_dataset: Dataset,
        args,
        collate_fn: DataCollator = DefaultDataCollator(return_tensors="pt")
) -> (DataLoader, DataLoader):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataloader, eval_dataloader
