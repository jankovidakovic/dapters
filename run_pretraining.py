import logging

import mlflow
import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader
from transformers import set_seed, DataCollatorForLanguageModeling, AutoModelForMaskedLM
import torch

from src.cli.pretraining import PreTrainingArguments, parse_args
from src.data import setup_dataloaders
from src.preprocess import pretraining_pipeline, hf_map, to_hf_dataset, sequence_columns, convert_to_torch
from src.trainer import train, evaluate_pretraining, pretraining_loss
from src.utils import setup_logging, maybe_tf32, get_tokenizer, get_tokenization_fn, pipeline, setup_optimizers

logger = logging.getLogger(__name__)


def main():
    # setup logging
    args: PreTrainingArguments = parse_args()
    setup_logging(args)
    set_seed(args.random_seed)
    maybe_tf32(args)

    tokenizer = get_tokenizer(args)

    do_tokenize = get_tokenization_fn(
        tokenizer=tokenizer,
        padding=args.padding,  # noqa
        truncation=True,
        max_length=args.max_length,
        return_special_tokens_mask=True,
        message_column=args.message_column
    )  # would be cool to abstract this also

    do_preprocess = pipeline(
        pd.read_csv,
        DataFrame.dropna,
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns)
    )

    dataset = do_preprocess(args.train_dataset_path)
    # eval_dataset = do_preprocess(args.eval_dataset_path)

    # seems to be working!!
    # train_dataloader, eval_dataloader = setup_dataloaders(
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=args.mlm_probability,
        )
    )

    # initialize model
    model = AutoModelForMaskedLM.from_pretrained(
        args.pretrained_model_name_or_path,
        cache_dir=args.cache_dir,
    )
    # we start from the model which is already pretrained

    # compile model
    model: PreTrainedModel = torch.compile(model).to(args.device)  # noqa

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
        epoch_steps=len(dataloader),
        scheduler_type=args.scheduler_type
    )   # TODO - dataloader was an IterableDataset, we wouldnt have len -> fix

    # set up mlflow
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
        train_dataloader=dataloader,
        epochs=args.epochs,
        # eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        # save_steps=args.save_steps,
        output_dir=args.output_dir,
        max_grad_norm=args.max_grad_norm,
        # early_stopping_patience=args.early_stopping_patience,
        # metric_for_best_model=args.metric_for_best_model,
        # greater_is_better=args.greater_is_better,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        get_loss=pretraining_loss(),
    )

    logger.warning("Training complete.")

    mlflow.end_run()


if __name__ == "__main__":
    main()