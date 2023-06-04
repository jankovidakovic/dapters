import logging

import hydra
import os.path
from pprint import pformat

import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DefaultDataCollator, AutoAdapterModel,
)

from src.distances.pairwise_distances import compute_pairwise_distances
from src.model_utils import setup_adapters, maybe_compile, set_device
from src.preprocess.steps import (
    multihot_to_list,
    to_hf_dataset,
    hf_map,
    convert_to_torch,
    sequence_columns,
)
from src.trainer import do_predict, compute_metrics
from src.types import Domain, DomainCollection
from src.utils import get_labels, get_tokenization_fn, pipeline

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="evaluate_adapter_finetuning")
def main(args: DictConfig):
    # TODO - we need to be able to load the adapter setup as well
    #   TODO -> just reuse hydra config

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    logger.info(OmegaConf.to_yaml(args))
    # load labels
    labels = get_labels(args.data.labels_path)

    mlflow.set_tracking_uri(args.mlflow.tracking_uri)
    mlflow.start_run(
        run_id=args.mlflow.run_id,
    )

    # we also need to be able to load the finetuned adapter from checkpoint

    # extract checkpoint step
    # checkpoint_step = checkpoint_name.split("/")[-1].split("-")[0]
    # since we assume responsibility to the caller, why not just accept checkpoint step efrom the caller

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
        model_max_length=64,
        do_lower_case=True,
    )

    do_tokenize = get_tokenization_fn(
        tokenizer=tokenizer,
        padding="max_length",
        truncation=True,
        max_length=64,
        message_column="preprocessed",
    )

    # load model
    if adapters_included := hasattr(args.model, "adapters"):
        model = AutoAdapterModel.from_pretrained(
            args.model.pretrained_model_name_or_path,
            cache_dir=args.model.cache_dir,
        )
        model = setup_adapters(model, args)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model.pretrained_model_name_or_path,
            cache_dir=args.model.cache_dir,
            problem_type="multi_label_classification",
            num_labels=args.data.num_labels
        )

    model = maybe_compile(model, args)
    model = set_device(model, args)

    model = model.to("cuda")  # TODO - device

    do_preprocess = pipeline(
        pd.read_csv,
        multihot_to_list(label_columns=labels, result_column="labels"),
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns),
    )

    source_dataset = do_preprocess(args.data.source_dataset_path)
    target_dataset = do_preprocess(args.data.target_dataset_path)

    source_dataloader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=DefaultDataCollator(return_tensors="pt"),
    )

    target_dataloader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=DefaultDataCollator(return_tensors="pt"),
    )

    source_predictions, source_references, source_hidden_states = do_predict(
        model, source_dataloader, output_hidden_states=True)
    target_predictions, target_references, target_hidden_states = do_predict(
        model, target_dataloader, output_hidden_states=True)

    source_metrics = compute_metrics(source_predictions, source_references, "source")
    logger.warning(pformat(source_metrics))
    mlflow.log_metrics(source_metrics, step=int(args.checkpoint_step))

    target_metrics = compute_metrics(target_predictions, target_references, "target")
    logger.warning(pformat(target_metrics))
    mlflow.log_metrics(target_metrics, step=int(args.checkpoint_step))

    # now we obtain domain shift!

    source_domain = Domain(name="source", representations=source_hidden_states, cluster_ids=None)
    target_domain = Domain(name="target", representations=target_hidden_states, cluster_ids=None)

    domain_collection = DomainCollection(
        domains=[source_domain, target_domain], pca_dim=args.pca_dim
    )  # its maybe because I increased the variance??

    domain_distances = compute_pairwise_distances(domain_collection)
    logger.warning(pformat(domain_distances))

    mlflow.log_metrics(
        domain_distances["centroid_distances"], step=int(args.checkpoint_step)
    )
    mlflow.log_metrics(
        domain_distances["domain_divergence_metrics"], step=int(args.checkpoint_step)
    )

    # this should actually work now, no?

    mlflow.end_run()


if __name__ == "__main__":
    main()
