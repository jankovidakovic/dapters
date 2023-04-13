import logging

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    set_seed,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer
)

from src.data import setup_dataloaders
from src.cli.finetuning import FineTuningArguments, parse_args
from src.preprocess import fine_tuning_pipeline
from src.trainer import train, evaluate_finetuning, fine_tuning_loss
from src.utils import setup_logging, get_labels, get_tokenization_fn, setup_optimizers, maybe_tf32, get_tokenizer, \
    pipeline, get_label_converter

import mlflow

logger = logging.getLogger(__name__)


def main():
    args: FineTuningArguments = parse_args()
    setup_logging(args)

    set_seed(args.random_seed)

    maybe_tf32(args)

    tokenizer = get_tokenizer(args)

    tokenization_fn = get_tokenization_fn(
        tokenizer=tokenizer,
        padding=args.padding,  # noqa
        truncation=True,
        max_length=args.max_length,
    )
    labels = get_labels(args.labels_path)

    do_preprocess = pipeline(
        pd.read_csv,
        fine_tuning_pipeline(
            initial_preprocessing=args.preprocessing,
            tokenizing_fn=do_tokenize,
            label_converter=get_label_converter(labels)
        )
    )

    train_dataset = do_preprocess(args.train_dataset_path)
    eval_dataset = do_preprocess(args.eval_dataset_path)

    # TODO - IterableDataset = ?
    train_dataloader, eval_dataloader = setup_dataloaders(
        train_dataset, eval_dataset, args
    )

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=len(labels),
        problem_type=args.problem_type,
        cache_dir=args.cache_dir,
    )
    model = torch.compile(model).to(args.device)  # noqa
    # TODO - look into torch.compile options

    logger.info(f"Model loaded successfully on device: {model.device}")

    # TODO - make configurable
    optimizer, scheduler = setup_optimizers(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_percentage=args.warmup_percentage,
        epochs=args.epochs,
        epoch_steps=len(train_dataloader),
        scheduler_type=args.scheduler_type
    )   # TODO - dataloader was an IterableDataset, we wouldnt have len -> fix

    # set up mlflow
    mlflow.set_tracking_uri("http://localhost:34567")
    mlflow.set_experiment(args.mlflow_experiment)

    mlflow.log_params(vars(args))

    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=args.epochs,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        label_names=labels,
        evaluation_threshold=args.evaluation_threshold,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    logger.warning("Training complete.")

    mlflow.end_run()


if __name__ == "__main__":
    main()

    # okay so MLFlow is basically:
    #   experiment_name -> e.g. "full fine-tuning"
    #   run_name -> e.g.        "fine-tuning on source with hparams x,y,z"
    #
    #   or is it better to have:
    #   experiment_name "fine-tuning on source" and then
    #   run_name "this hparams" -> im not actually sure
    #
    # log:
    #   model
    #   hyperparameters (or just a config file)
    #   results (metrics) -> metrics arent really trivial to log from only training
    #
    #   I should be able to track domain shift while running, no?
    #       -> this could be done using a separate process
    #       -> which basically evaluates the model asynchronously (but that might suck idk)
    #
    # TODO:
    # 1. define exactly what experiments will be running
    # 2.
    #
    # this is basically systems design
    # 1. loading csv file, preprocessing it, saving the processed csv file
    # 2. loading one csv file, performing splitting, saving to multiple csv files
    # 3. train model, given some configuration and some dataset
    # 4. evaluate model, given some model file and some dataset (right?)
