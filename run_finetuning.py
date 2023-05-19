import logging
import os
from functools import partial

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from transformers import (
    set_seed,
)

from src.models import setup_model
from src.preprocess.steps import multihot_to_list, to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from src.trainer import train, fine_tuning_loss, evaluate_finetuning
from src.utils import get_labels, get_tokenization_fn, setup_optimizers, maybe_tf32, get_tokenizer, \
    pipeline, mean_binary_cross_entropy, save_adapter_model, save_transformer_model

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="finetuning")
def main(args: DictConfig):
    # lets see if this actually works now
    # args = parse_args("finetuning")
    logger.info(OmegaConf.to_yaml(args))

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # removed setup_logging because hydra does that for me

    set_seed(args.random_seed)  # oh but this actually already works because argparse has __getattr__ implemented

    maybe_tf32(args)

    tokenizer = get_tokenizer(args)

    do_tokenize = get_tokenization_fn(
        tokenizer=tokenizer,
        padding=args.padding,  # noqa
        truncation=True,
        max_length=args.max_length,
        message_column=args.message_column
    )
    labels = get_labels(args.labels_path)

    do_preprocess = pipeline(
        pd.read_csv,
        DataFrame.dropna,
        multihot_to_list(
            label_columns=labels,
            result_column="labels"
        ),
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns)
    )

    train_dataset = do_preprocess(args.data.train_dataset_path)
    eval_dataset = do_preprocess(args.data.eval_dataset_path)

    model = setup_model(args)  # works both with adapters and non-adapters

    epoch_steps = len(train_dataset) // args.per_device_train_batch_size // args.gradient_accumulation_steps

    optimizer, scheduler = setup_optimizers(
        model,
        lr=args.optimizer.learning_rate,
        weight_decay=args.optimizer.weight_decay,
        adam_epsilon=args.optimizer.adam_epsilon,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_percentage=args.optimizer.warmup_percentage,
        epochs=args.epochs,
        epoch_steps=epoch_steps,
        scheduler_type=args.optimizer.scheduler_type
    )

    if use_mlflow := hasattr(args, "mlflow"):
        import mlflow
        mlflow.set_tracking_uri(args.mlflow.tracking_uri)
        mlflow_experiment = mlflow.set_experiment(args.mlflow.experiment)

        mlflow.start_run(
            experiment_id=mlflow_experiment.experiment_id,
            run_name=args.mlflow.run_name,
            description=args.mlflow.run_description
        )

        # aha i can actually log to hydra, right?

        # log run_id to logging file, to be found later by grep
        logger.info(f"MLFlow run_id={mlflow.active_run().info.run_id}")

        # predivno
        mlflow.log_params(pd.json_normalize(OmegaConf.to_container(args, resolve=True), sep=".").to_dict(orient="records")[0])

        # set the tags
        mlflow.set_tags(args.mlflow.tags)

    # this can be "with maybe_mlflow(args)"
    # or it can be a decorator

    train(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        do_evaluate=evaluate_finetuning(
            evaluation_threshold=args.evaluation_threshold,
        ),
        get_loss=fine_tuning_loss(loss_fn=mean_binary_cross_entropy),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mlflow=use_mlflow,
        evaluate_on_train=args.evaluate_on_train,
        dataloader_num_workers=args.dataloader_num_workers,
        model_saving_callback=partial(save_adapter_model, adapter_name=args.adapters.adapter_name)
        if hasattr(args, "adapters")
        else save_transformer_model
    )

    logger.warning("Training complete.")

    if use_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
