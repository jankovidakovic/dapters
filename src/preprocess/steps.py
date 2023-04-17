import logging
from typing import Callable, Optional

import pandas as pd
import torch.utils.data
from datasets import Dataset, Sequence

from src.utils import pipeline

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


def keep(columns: list[str]):
    def apply(df):
        logger.warning(f"Keeping columns: {columns}")
        return df.loc[:, columns]
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


def log_size(dataset: Dataset):
    logger.warning(f"Dataset size: {len(dataset)}")
    return dataset


def sequence_columns(dataset: Dataset) -> list[str]:
    return list(map(
        lambda entry: entry[0],  # feature key (name)
        filter(
            lambda entry: isinstance(entry[1], Sequence),  # value is a Sequence
            dataset.features.items())))


def sample_by(sample_size: int, group_by: Optional[str | list[str]] = None) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def apply(df: pd.DataFrame):
        logger.warning(f"Sampling {sample_size} examples" + (f" by {group_by}" if group_by else ""))
        if group_by:
            return df.groupby(group_by).sample(sample_size)
        else:
            return df.sample(sample_size)

    return apply


dummy = pipeline(
    drop(columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags"]),
    sample_by(1, group_by="cluster_id"),
    deduplication("message"),
    drop(["cluster_id"])
)


fine_tuning_dev = pipeline(
    drop(columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags", "cluster_id"]),
    deduplication("message")
)

