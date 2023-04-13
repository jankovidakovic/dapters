import logging
from typing import Callable

import pandas as pd
import torch.utils.data
from datasets import Dataset, Sequence

from src.utils import pipeline, sample_by

logger = logging.getLogger(__name__)


def deduplication(subset: str | list[str]):
    def apply(df):
        logger.warning(f"Dropping duplicates by subset: {subset}")
        return df.drop_duplicates(subset=subset)
    return apply


def drop(columns: list[str]):
    def apply(df):
        logger.warning(f"Dropping columns: {columns}")
        return df.drop(columns=columns)
    return apply


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    logger.warning(f"Converting to HF dataset")
    return Dataset.from_pandas(df, preserve_index=False)


def hf_map(mapping_fn: Callable, batched: bool = False):
    def apply(dataset: Dataset):
        logger.warning(f"Mapping dataset using {mapping_fn.__name__}")
        return dataset.map(mapping_fn, batched=batched)
    return apply


def convert_to_torch(columns: list[str] | Callable[[Dataset], list[str]]) -> Callable[[Dataset], torch.utils.data.Dataset]:
    def apply(dataset: Dataset) -> torch.utils.data.Dataset:
        logger.warning(
            f"Converting to PyTorch dataset, "
            f"keeping the following_columns: "
            f"{columns if isinstance(columns, list) else columns.__name__}")
        return dataset.with_format(
            "torch",
            columns=columns if isinstance(columns, list) else columns(dataset)  # noqa
        )
    return apply


def sequence_columns(dataset: Dataset) -> list[str]:
    return list(map(
        lambda entry: entry[0],  # feature key (name)
        filter(
            lambda entry: isinstance(entry[1], Sequence),  # value is a Sequence
            dataset.features.items())))

dummy = pipeline(
    drop(columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags"]),
    sample_by("cluster_id", 1),
    deduplication("message"),
    drop(["cluster_id"])
)


fine_tuning_dev = pipeline(
    drop(columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags", "cluster_id"]),
    deduplication("message")
)

