from typing import Callable

from .steps import hf_map, logger, to_hf_dataset, convert_to_torch, sequence_columns
from .steps import dummy, fine_tuning_dev, logger, to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from ..utils import dynamic_import, pipeline


def fine_tuning_pipeline(
        initial_preprocessing: str,   # dynamic import
        tokenizing_fn: Callable,
        label_converter: Callable
):
    preprocessing = dynamic_import("src.preprocess", initial_preprocessing)
    logger.warning(f"Using preprocessing method: {initial_preprocessing}")

    return pipeline(
        preprocessing,
        to_hf_dataset,
        hf_map(tokenizing_fn, batched=True),
        hf_map(label_converter, batched=False),
        convert_to_torch(columns=sequence_columns)
    )


def pretraining_pipeline(
        initial_preprocessing: str,   # dynamic import
        tokenizing_fn: Callable,
):
    preprocessing = dynamic_import("src.preprocess", initial_preprocessing)
    logger.warning(f"Using preprocessing method: {initial_preprocessing}")

    return pipeline(
        preprocessing,
        to_hf_dataset,
        hf_map(tokenizing_fn, batched=True),
        convert_to_torch(columns=sequence_columns)
    )
