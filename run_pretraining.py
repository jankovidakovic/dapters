import logging
import os

import hydra
import pandas as pd
from math import ceil
from omegaconf import OmegaConf, DictConfig
from pandas import DataFrame
from transformers import set_seed, DataCollatorForLanguageModeling, AutoModelForMaskedLM, AutoAdapterModel

from src.models import maybe_compile, set_device
from src.models.bottleneck_adapters import setup_adapter_pretraining
from src.preprocess import hf_map, to_hf_dataset, sequence_columns, convert_to_torch
from src.trainer import train, pretraining_loss, evaluate_pretraining
from src.utils import maybe_tf32, get_tokenizer, get_tokenization_fn, pipeline, setup_optimizers, get_adapter_saver, \
    save_transformer_model

logger = logging.getLogger(__name__)

# in theory, this should work with adapters as well, right?


@hydra.main(version_base="1.3", config_path="configs", config_name="pretraining")
def main(args: DictConfig):
    # setup logging
    logger.info(OmegaConf.to_yaml(args))

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(args.random_seed)
    maybe_tf32(args)

    tokenizer = get_tokenizer(args.model)

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

    train_dataset = do_preprocess(args.data.train_dataset_path)
    eval_dataset = do_preprocess(args.data.eval_dataset_path)

    # initialize model

    if is_adapter_pretraining := hasattr(args.model, "adapter"):
        model = AutoAdapterModel.from_pretrained(
            args.pretrained_model_name_or_path,
            cache_dir=args.cache_dir,
        )
        # we start from the model which is already pretrained

        model = setup_adapter_pretraining(model, args.model.adapter)
        model.add_masked_lm_head(args.adapter_name)
        model.train_adapter(args.adapter_name)
        logger.warning(model.adapter_summary())

    else:
        model = AutoModelForMaskedLM.from_pretrained(
            args.pretrained_model_name_or_path,
            cache_dir=args.cache_dir,
        )

    model = maybe_compile(model, args)
    model = set_device(model, args)

    logger.info(f"Model loaded successfully on device: {model.device}")

    epoch_steps = ceil(len(train_dataset) / args.training.per_device_train_batch_size / args.training.gradient_accumulation_steps)

    # TODO - make configurable
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

    train(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=args.mlm_probability,
        ),
        per_device_train_batch_size=args.training.per_device_train_batch_size,
        per_device_eval_batch_size=args.training.per_device_eval_batch_size,
        epochs=args.training.epochs,
        max_grad_norm=args.training.max_grad_norm,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        get_loss=pretraining_loss(),
        do_evaluate=evaluate_pretraining(),
        use_mlflow=use_mlflow,
        dataloader_num_workers=args.training.dataloader_num_workers,
        model_saving_callback=get_adapter_saver(args.model.adapter.name) if is_adapter_pretraining else save_transformer_model
    )

    logger.warning("Training complete.")

    if use_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
