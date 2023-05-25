import logging
import os
from functools import partial

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from transformers import (
    set_seed, AutoAdapterModel, AutoModelForSequenceClassification,
)

from src.model_utils import setup_adapters, maybe_compile, set_device
from src.preprocess.steps import multihot_to_list, to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from src.trainer import train, fine_tuning_loss, evaluate_finetuning
from src.utils import get_labels, get_tokenization_fn, setup_optimizers, maybe_tf32, get_tokenizer, \
    pipeline, mean_binary_cross_entropy, save_adapter_model, save_transformer_model, get_adapter_saver

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
        padding=args.tokenizer.padding,  # noqa
        truncation=True,
        max_length=args.tokenizer.max_length,
        message_column=args.data.message_column
    )
    labels = get_labels(args.data.labels_path)

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

    # TODO - fix this for "domain adaptation"

    epoch_steps = len(train_dataset) // args.training.per_device_train_batch_size // args.training.gradient_accumulation_steps
    optimizer, scheduler = setup_optimizers(
        model,
        lr=args.optimizer.learning_rate,
        weight_decay=args.optimizer.weight_decay,
        adam_epsilon=args.optimizer.adam_epsilon,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        warmup_percentage=args.optimizer.warmup_percentage,
        epochs=args.training.epochs,
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
        per_device_train_batch_size=args.training.per_device_train_batch_size,
        per_device_eval_batch_size=args.training.per_device_eval_batch_size,
        epochs=args.training.epochs,
        max_grad_norm=args.training.max_grad_norm,
        do_evaluate=evaluate_finetuning(
            evaluation_threshold=args.training.evaluation_threshold,
        ),
        get_loss=fine_tuning_loss(loss_fn=mean_binary_cross_entropy),
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        use_mlflow=use_mlflow,
        evaluate_on_train=args.training.evaluate_on_train,
        dataloader_num_workers=args.training.dataloader_num_workers,
        model_saving_callback=get_adapter_saver("finetuning")
        if adapters_included
        else save_transformer_model
    )

    logger.warning("Training complete.")

    if use_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
