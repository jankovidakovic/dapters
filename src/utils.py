import json
import operator
from functools import partial
from pprint import pformat
from typing import Callable
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel, get_scheduler, AutoTokenizer, BatchEncoding
from transformers.utils import PaddingStrategy

from src.types import HiddenRepresentationConfig, Domain

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


def make_logfile_name(args):
    return args.log_path


def setup_logging(args):
    # logging
    handlers = [logging.StreamHandler()]
    if hasattr(args, "log_path"):
        dirname = os.path.dirname(os.path.abspath(args.log_path))
        os.makedirs(dirname, exist_ok=True)
        log_filename = make_logfile_name(args)
        handlers.append(logging.FileHandler(filename=log_filename, mode="w"))
        logging.info(f"Logging to file: {os.path.abspath(log_filename)}")

    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # log info only on main process
        level=logging.INFO,  # TODO - info only if verbose?
        handlers=handlers,
    )


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
    # labels = sorted(labels["labels"])
    labels = labels["labels"]  # lets not sort labels anymore

    logger.info(f"labels = {pformat(labels)}")

    return labels


def get_tokenization_fn(
        tokenizer: PreTrainedTokenizer,
        padding: PaddingStrategy = PaddingStrategy.LONGEST,
        truncation: bool = True,
        max_length: int = 64,
        return_special_tokens_mask: bool = False,
        message_column: str = "message"
):
    def tokenize(examples):
        return tokenizer(
            examples[message_column],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=return_special_tokens_mask
        )

    return tokenize


def save_transformer_model(model, output_dir):
    model.save_pretrained(output_dir)


def save_adapter_model(model, output_dir, adapter_name):
    model.save_adapter(
        save_directory=output_dir,
        adapter_name=adapter_name,
        with_head=True
    )


def save_checkpoint(
        model: PreTrainedModel,
        output_dir: str,
        global_step: int,
        tokenizer: PreTrainedTokenizer,
        use_mlflow: bool = False,
        model_saving_callback: Callable = save_transformer_model
):
    checkpoint_name = f"checkpoint-{global_step}"
    output_dir = os.path.join(output_dir, checkpoint_name)  # moze
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model_saving_callback(model, output_dir)
    logger.info(f"Saved model checkpoint to {output_dir}")

    tokenizer.save_pretrained(output_dir)
    logger.warning(f"Saved tokenizer to {output_dir}")

    if use_mlflow:
        import mlflow
        mlflow.log_artifacts(output_dir, artifact_path=checkpoint_name)


def dynamic_import(
        module_name: str = "data_preprocessing_pipeline",
        function_name: str = "dummy_preprocessing_one"
):
    """ Loads a function from a module.

    :param module_name: name of the module.
    :param function_name: name of the function.
    :return: function object.
    """
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def is_improved(
        current_value: float,
        best_value: float,
        greater_is_better: bool
):
    compare = operator.gt if greater_is_better else operator.lt
    return compare(current_value, best_value)


def setup_optimizers(
        model: nn.Module,
        lr: float,
        weight_decay: float,
        adam_epsilon: float,
        epochs: int,
        gradient_accumulation_steps: int,
        warmup_percentage: float,
        epoch_steps: int,
        scheduler_type: str
) -> (Optimizer, LRScheduler):
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_epsilon,
        capturable=True,  # make optimizer capturable in a CUDA graph
        fused=True  # fuse operations into a single CUDA kernel (TODO - might break on TF32)
    )

    # calculate the warmup steps
    num_training_steps = epochs * epoch_steps
    logger.warning(f"Total steps = {num_training_steps}")
    num_warmup_steps = round(warmup_percentage * num_training_steps)
    logger.warning(f"Warmup percentage was set to {warmup_percentage}. "
                   f"Based on that, warmup will be performed over {num_warmup_steps} warmup steps.")

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler


def maybe_tf32(args):
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True  # noqa
        torch.backends.cudnn.allow_tf32 = True  # noqa
        logger.warning("TF32 enabled.")


def get_tokenizer(args) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        model_max_length=args.max_length,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )


def set_device(
        batch: BatchEncoding,
        device: torch.device
) -> BatchEncoding:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def get_cls_token(hidden_states: torch.Tensor) -> torch.Tensor:
    # batch_size, sequence_length, hidden_size = hidden_states.shape
    return hidden_states[:, 0, :]


def get_representations(
        model: nn.Module,
        dataloader: DataLoader,
        n_examples: int,
        semantic_composition: Callable[[torch.Tensor], torch.Tensor] = get_cls_token
):
    representations = np.empty((n_examples, model.config.hidden_size))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference"):
            set_device(batch, model.device)
            hidden_states = model(**batch).last_hidden_state

            slice_index = slice(i * dataloader.batch_size, (i+1) * dataloader.batch_size)
            representations[slice_index, :] = semantic_composition(hidden_states).detach().cpu().numpy()

        # TODO - make this layer by layer

    return representations


def get_domain_from_config(
        config: HiddenRepresentationConfig
) -> Domain:
    logger.warning(f"Initializing domain {config.name}")
    # load cluster ids
    df = pd.read_csv(config.processed_datasets[0])
    cluster_ids = df.loc[:, "cluster_id"].to_numpy()
    logger.warning(f"Loaded {len(cluster_ids)} cluster_ids")

    # load hidden representations
    representations = np.load(config.cls_representations[0])
    logger.warning(f"Loaded representations of shape {representations.shape}")

    # compute centroid
    centroid = np.mean(representations, axis=0)
    logger.warning(f"Computed representation centroid.")

    return Domain(
        representations=representations,
        centroid=centroid,
        cluster_ids=cluster_ids,
        name=config.name
    )


def mean_binary_cross_entropy(
        input: torch.Tensor,
        target: torch.Tensor
):
    return F.binary_cross_entropy_with_logits(input, target, reduction="mean")