import logging
import os

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AdapterConfig


logger = logging.getLogger(__name__)


def maybe_compile(model, args):
    if args.use_torch_compile:
        logger.warning(f"Compiling the model")
        model = torch.compile(model)
        logger.warning(f"Model successfully compiled.")
    return model


def set_device(model, args):
    model = model.to(args.device)
    logger.info(f"Model loaded successfully on device: {model.device}")
    return model


def setup_adapters(model, args: DictConfig, log_fn=logger.info):
    """ Sets up any adapters specified in the config
    Returns the fully set up model

    :param model:
    :param args: config. Keys marked with ? are optional. Relevant config shape:

        model:
            adapters:
                (pretraining? | finetuning?):
                    {pretrained_path: str} | {config: str, reduction_factor?: int}
                    with_head: bool
                train:
                    - (pretraining? | finetuning?)

    :param log_fn: logging function
    :return: model
    """

    if adapters_included := hasattr(args.model, "adapters"):
        log_fn(f"Setting up adapters from config: ")
        log_fn(OmegaConf.to_yaml(args.model.adapters))

        num_heads = 0
        adapter_setup = []

        # first, setup the pretraining adapter
        if adapter_pretraining := hasattr(args.model.adapters, "pretraining"):
            log_fn(f"Setting up pretraining adapter with the following config: ")
            log_fn(OmegaConf.to_yaml(args.model.adapters.pretraining))
            if hasattr(args.model.adapters.pretraining, "pretrained_path"):
                log_fn(
                    f"Loading pretrained adapter from {os.path.abspath(args.model.adapters.pretraining.pretrained_path)}")
                model.load_adapter(args.model.adapters.pretraining.pretrained_path,
                                   load_as="pretraining", with_head=args.model.adapters.pretraining.with_head)
                if args.model.adapters.pretraining.with_head:
                    num_heads += 1
            else:
                log_fn(f"No pretrained adapter specified for pretraining, initializing a new adapter")
                model.add_adapter(adapter_name="pretraining",
                                  config=AdapterConfig.load(
                                      args.model.adapters.pretraining.config,
                                      reduction_factor=args.model.adapters.pretraining.reduction_factor
                                  ))
                if args.model.adapters.pretraining.with_head:
                    log_fn(f"Adding a masked language modeling head to the adapter")
                    model.add_masked_lm_head("pretraining")  # masterchef
                    num_heads += 1
            adapter_setup.append("pretraining")
            # okay this seems to work for pretrained adapters, lets TEST
        # load finetuning adapter
        if adapter_finetuning := hasattr(args.model.adapters, "finetuning"):
            log_fn(f"Setting up finetuning adapter with the following config: ")
            log_fn(OmegaConf.to_yaml(args.model.adapters.finetuning))
            if hasattr(args.model.adapters.finetuning, "pretrained_path"):
                log_fn(
                    f"Loading pretrained adapter from {os.path.abspath(args.model.adapters.finetuning.pretrained_path)}")
                model.load_adapter(args.model.adapters.finetuning.pretrained_path,
                                   load_as="finetuning", with_head=args.model.adapters.finetuning.with_head)
                if args.model.adapters.finetuning.with_head:
                    log_fn(f"Loaded head for finetuning adapter from pretrained checkpoint")
                    num_heads += 1
            else:
                log_fn(f"No pretrained adapter specified for finetuning, initializing a new adapter")
                model.add_adapter(adapter_name="finetuning",
                                  config=AdapterConfig.load(
                                      args.model.adapters.finetuning.config,
                                      reduction_factor=args.model.adapters.finetuning.reduction_factor
                                  ))
                if args.model.adapters.finetuning.with_head:
                    log_fn(f"Adding a classification head to the adapter")
                    if num_heads > 0:
                        log_fn("Head count error")
                        raise RuntimeError("A head is already loaded! this should probably not happen, exiting.")
                    model.add_classification_head("finetuning", num_labels=args.model.num_labels, multilabel=True)
                    num_heads += 1
            adapter_setup.append("finetuning")

        if train_adapters := hasattr(args.model.adapters, "train"):
            log_fn(f"Training the following adapters: ")
            log_fn(OmegaConf.to_yaml(args.model.adapters.train))
            model.train_adapter(args.model.adapters.train)

        model.set_active_adapters(adapter_setup)

        log_fn(f"{num_heads = }")

        log_fn(model.adapter_summary())

    else:
        log_fn(f"No adapters specified in config, skipping")

    return model
