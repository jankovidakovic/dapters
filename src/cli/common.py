from argparse import ArgumentParser


def add_configuration_args(parser: ArgumentParser):
    group = parser.add_argument_group("configuration", "Configuration arguments")

    group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training. Defaults to cuda."
    )

    group.add_argument(
        "--use_tf32",
        action="store_true",
        default=True,
        help="Whether to use TensorFloat32. Defaults to True."
    )

    group.add_argument(
        "--use_torch_compile",
        action="store_true",
        default=False,
        help="If set, model will be compiled using torch.compile()."
    )

    group.add_argument(
        "--random_seed",
        type=int,
        default=192837465,
        help="Random seed. Defaults to 192837465."
    )  # TODO

    group.add_argument(
        "--mlflow_experiment",
        type=str,
        default=None,
        help="Name of the MLFlow experiment. Defaults to None."
    )  # TODO

    group.add_argument(
        "--log_path",
        type=str,
        default="./run.log",
        help="Path to the log file. Defaults to ./run.log."
    )

    # directories relevant for IO
    group.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./output/test",
        help="The output directory where the model predictions and "
             "checkpoints will be written."
    )

    group.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:34567",
        help="MLFlow tracking URI. Defaults to http://localhost:34567"
    )

    group.add_argument(
        "--mlflow_run_name",
        type=str,
        default=None,
        help="MLFlow run name. Defaults to an empty string."
    )

    group.add_argument(
        "--mlflow_run_description",
        type=str,
        default="Generic MLFlow run description.",
        help="MLFlow run description. Defaults to an empty string."
    )

    return parser


def add_tokenizer_args(parser: ArgumentParser):
    group = parser.add_argument_group(
        "Tokenizer arguments",
        "Tokenizer arguments"
    )

    group.add_argument(
        "--padding",
        type=str,
        choices=["longest", "max_length", "do_not_pad"],
        default="max_length",
        help="Padding strategy when tokenizing. Defaults to 'max_length'."
    )

    group.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Model max length. Defaults to 64."
    )

    group.add_argument(
        "--truncation",
        action="store_true",
        default=True,
        help="If set, will truncate sequences for which length exceeds --max_length."
    )

    group.add_argument(
        "--do_lower_case",
        action="store_true",
        help="If set, input text will be lowercased during tokenization. "
                "This flag is useful when one is using uncased models (e.g. 'bert-base-uncased')",
    )

    group.add_argument(
        "--cache_dir",
        type=str,
        help="Huggingface cache directory."
    )


def add_model_args(parser: ArgumentParser):

    group = parser.add_argument_group(
        "Model arguments",
        "Arguments relevant for the initialization of Huggingface model."
    )

    group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="Fully qualified model name, either on Huggingface Model Hub or "
             "a local filesystem path."
    )


def add_optimizer_args(parser: ArgumentParser):
    # add optimizer argument group
    optimizer_group = parser.add_argument_group(
        "optimizer",
        "Arguments related to the optimizer."
    )

    # add optimizer arguments
    optimizer_group.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate used for optimization. Defaults to 2e-5.",
    )
    optimizer_group.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Stability factor used in ADAM Optimizer, used to mitigate zero-division errors. "
             "Defaults to 1e-8.",
    )
    optimizer_group.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay factor. Defaults to 0 (no weight decay)."
    )
    optimizer_group.add_argument(
        "--scheduler_type",
        default="constant",
        choices=["linear", "constant", "cosine", "polynomial", "constant_with_warmup"],
        type=str,
        help="The type of scheduler to use. Defaults to constant.",
    )

    optimizer_group.add_argument(
        "--warmup_percentage",
        type=float,
        default=0,
        help="Percentage of training for which the linear warmup will be used. Defaults to 0, which means no warmup."
    )


def add_data_args(parser):
    group = parser.add_argument_group("data", "Arguments relevant for the data.")
    group.add_argument(
        "--train_dataset_path", type=str, help="Filesystem path to a training dataset."
    )

    group.add_argument(
        "--eval_dataset_path",
        type=str,
        help="Filesystem path to the evaluation dataset.",
        default=None
    )  # this is actually the "early stopping" split.

    group.add_argument(
        "--message_column",
        type=str,
        default="preprocessed",
        help="Name of the column containing the message to be classified."
    )


def add_training_args(parser: ArgumentParser):
    group = parser.add_argument_group("training", "Training arguments")

    group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device. Defaults to 4."
    )

    group.add_argument(
        "--per_device_eval_batch_size",
        default=2,
        type=int,
        help="Batch size used during evaluation (per device). Defaults to 2.",
    )

    group.add_argument(
        "--max_grad_norm",
        default=None,
        type=float,
        help="Maximum value of L2-norm of the gradients during optimization. Gradients "
             "with norm greater than this value will be clipped. Defaults to None"
    )

    group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers. Defaults to 4."
    )

    group.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs. Defaults to 1."
    )

    group.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of steps to perform between each loss logging."
    )

    group.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Period of model and tokenizer saving (in steps). "
             "Model and tokenizer are also saved at the end of each epoch."
             "If 'save_steps' is not specified, then no additional saving (besides the end of epoch) is done.",
    )

    group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of steps to wait before early stopping. Defaults to None."
    )

    group.add_argument(
        "--metric_for_best_model",
        type=str,
        default=None,
        help="Metric to use for early stopping. Defaults to None."
    )

    group.add_argument(
        "--greater_is_better",
        action="store_true",
        default=False,
        help="Whether the metric for best model is considered better when greater or lower. Defaults to True."
    )

    group.add_argument(
        "--early_stopping_start",
        type=int,
        default=0,
        help="Number of epochs to wait before starting early stopping. Defaults to 0."
    )

    group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps. Defaults to 1."
    )

    group.add_argument(
        "--evaluate_on_train",
        action="store_true",
        default=False,
        help="Whether to evaluate on the training set (after every epoch). Defaults to False."
    )


def add_adapter_args(parser):
    group = parser.add_argument_group("adapter", "Arguments relevant for adapter training.")

    group.add_argument(
        "--adapter_config",
        type=str,
        choices=["pfeiffer", "houlsby"],
        default="pfeiffer",
        help="Adapter configuration to use. Defaults to 'pfeiffer'."
    )

    group.add_argument(
        "--reduction_factor",
        type=int,
        default=16,
        help="Reduction factor for the adapter. Defaults to 16."
    )

    group.add_argument(
        "--adapter_name",
        type=str,
        help="Name of the adapter to train. Defaults to None."
    )

    group.add_argument(
        "--pretrained_adapter_path",
        type=str,
        required=False,
        help="If provided, pretrained (language) adapter will be loaded from the given path."
             "During training, adapters will be stacked, and only the fine-tuning adapter will be trained."
    )

    # thats everything