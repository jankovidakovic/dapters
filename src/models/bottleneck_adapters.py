import logging
import os

from omegaconf import DictConfig
from transformers import AutoAdapterModel, AdapterConfig

logger = logging.getLogger(__name__)


def get_adapter_model(args: DictConfig):
    # initialize model
    model = AutoAdapterModel.from_pretrained(
        args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
        problem_type="multi_label_classification"
    )

    adapter_config = AdapterConfig.load(args.adapters.adapter_config, reduction_factor=args.adapters.reduction_factor)

    logger.warning(f"Loaded the following adapter config: {adapter_config}")
    model.add_adapter(args.adapters.adapter_name, adapter_config)
    model.add_classification_head(args.adapters.adapter_name, num_labels=args.num_labels, multilabel=True)

    if hasattr(args.adapters, "pretrained_adapter_path"):
        logger.warning(f"Loading pretrained adapter from {os.path.abspath(args.adapters.pretrained_adapter_path)}")
        model.load_adapter(args.adapters.pretrained_adapter_path, load_as="pt")

    model.train_adapter(args.adapters.adapter_name)   # training the fine-tuning adapter)
    logger.warning(f"Model state set to training of adapter: {args.adapters.adapter_name}")
    if hasattr(args.adapters, "pretrained_adapter_path"):
        model.set_active_adapters(["pt", args.adapters.adapter_name])
    else:
        model.set_active_adapters([args.adapters.adapter_name])

    logger.warning(model.adapter_summary())

    return model
