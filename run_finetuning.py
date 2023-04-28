import logging

import pandas as pd
import torch
from pandas import DataFrame
from transformers import (
    set_seed,
    AutoModelForSequenceClassification,
)

from src.data import setup_dataloaders
from src.cli.finetuning import FineTuningArguments, parse_args
from src.preprocess.steps import multihot_to_list, to_hf_dataset, hf_map, convert_to_torch, sequence_columns
from src.trainer import train, fine_tuning_loss, evaluate_finetuning
from src.utils import setup_logging, get_labels, get_tokenization_fn, setup_optimizers, maybe_tf32, get_tokenizer, \
    pipeline, mean_binary_cross_entropy


logger = logging.getLogger(__name__)


def main():
    args: FineTuningArguments = parse_args()
    setup_logging(args)

    set_seed(args.random_seed)

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

    train_dataset = do_preprocess(args.train_dataset_path)
    eval_dataset = do_preprocess(args.eval_dataset_path)

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=len(labels),
        cache_dir=args.cache_dir,
        problem_type="multi_label_classification"
    )
    model = torch.compile(model)
    model = model.to(args.device)
    # TODO - look into torch.compile options

    logger.info(f"Model loaded successfully on device: {model.device}")

    epoch_steps = len(train_dataset) // args.per_device_train_batch_size

    optimizer, scheduler = setup_optimizers(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_percentage=args.warmup_percentage,
        epochs=args.epochs,
        epoch_steps=epoch_steps,
        scheduler_type=args.scheduler_type
    )

    # set up mlflow
    if use_mlflow := (args.mlflow_experiment is not None):
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow_experiment = mlflow.set_experiment(args.mlflow_experiment)

        mlflow.start_run(
            experiment_id=mlflow_experiment.experiment_id,
            run_name=args.mlflow_run_name,
            description=args.mlflow_run_description
        )

        mlflow.log_params(vars(args))



    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        max_grad_norm=args.max_grad_norm,
        do_evaluate=evaluate_finetuning(
            evaluation_threshold=args.evaluation_threshold,
        ),
        get_loss=fine_tuning_loss(loss_fn=mean_binary_cross_entropy),
        early_stopping_patience=args.early_stopping_patience,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mlflow=use_mlflow,
        evaluate_on_train=args.evaluate_on_train,
    )

    logger.warning("Training complete.")

    if use_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
