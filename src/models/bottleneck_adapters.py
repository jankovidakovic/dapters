import logging
import os
from typing import Any

from omegaconf import DictConfig
from transformers import AdapterConfig

logger = logging.getLogger(__name__)


# we need:
#   maybe add ft adapter -> also adds the classification head
#   maybe add pt adapter -> also adds the MLM head
#   maybe add stack      ->


def setup_adapter_finetuning(model, adapter_args, num_labels):
    """ Returns the model and the adapter setup

    :param model:
    :param adapter_args:
    :param num_labels:
    :return:
    """
    logger.warning(f"Adding a fine-tuning adapter")
    add_adapter(model, adapter_args)
    model.add_classification_head(adapter_args.name, num_labels=num_labels, multilabel=True)  # FUUUUCK

    return model, [adapter_args.name]


def add_adapter(model, adapter_args):
    """ Adds the adapter

    :param model:
    :param adapter_args:
    :return:
    """
    adapter_config = AdapterConfig.load(adapter_args.config, reduction_factor=adapter_args.reduction_factor)

    logger.warning(f"Loaded the following adapter config: {adapter_config}")
    model.add_adapter(adapter_args.config, adapter_config)

    return model

def setup_adapter_pretraining(model, adapter_args):
    """ Returns the model and the adapter setup

    :param model:
    :param adapter_args:
    :return:
    """
    logger.warning(F"Adding a pretraining adapter")
    add_adapter(model, adapter_args)
    model.add_masked_lm_head(adapter_args.name)

    return model, [adapter_args.name]


def get_adapter_model(model_args: DictConfig):
    # TODO
    model = Any  # lmao

    if adapter_pretraining := hasattr(model_args, "pretrained_adapter_path"):
        logger.warning(f"Loading pretrained adapter from {os.path.abspath(model_args.pretrained_adapter_path)}")
        model.load_adapter(model_args.pretrained_adapter_path, load_as="pretraining")

    model.train_adapter(model_args.adapter.name)   # training the fine-tuning adapter)
    logger.warning(f"Model state set to training of adapter: {model_args.adapter.name}")

    if adapter_pretraining:
        model.set_active_adapters(["pretraining", model_args.adapter.name])
    else:
        model.set_active_adapters([model_args.adapter.name])

    logger.warning(model.adapter_summary())

    raise NotImplementedError("ree")
    #  return model
