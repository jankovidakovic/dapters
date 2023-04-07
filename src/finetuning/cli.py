import argparse


def get_parser():
    parser = argparse.ArgumentParser("Fine-tuning arguments parser")

    parser.add_argument(
        "--train_dataset_path", type=str, help="Filesystem path to a training dataset."
    )

    parser.add_argument(
        "--eval_dataset_path", type=str, help="Filesystem path to the evaluation dataset."
    )  # this is actually the "early stopping" split.

    parser.add_argument(
        "--log_path",
        type=str,
        help="Filesystem path to a log file in which logs will be written.",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="Fully qualified model name, either on Huggingface Model Hub or "
             "a local filesystem path."
    )

    parser.add_argument(
        "--padding",
        type=str,
        choices=["longest", "max_length", "do_not_pad"],
        default="max_length",
        help="Padding strategy when tokenizing. Defaults to 'max_length'."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Model max length. Defaults to 64."
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        help="Path to a JSON file with the layout {'labels': ['LABEL_0', 'LABEL_1', ...]}"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device. Defaults to 4."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers. Defaults to 4."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs. Defaults to 1."
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of gradient steps to perform between each evaluation."
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of steps to perform between each loss logging."
    )

    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        required=True,
        help="Number of unique labels in dataset. If not provided, will "
             "be calculated as the number of unique values in labels column "
             "in the training dataset. Name of the column containig labels can "
             "be set using the '--label_column' option.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path. If not provided, will default to "
             "value of '--pretrained_model_name_or_path'.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path. If not provided, will default to "
             "value of '--pretrained_model_name_or_path'.",
    )

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="If set, input text will be lowercased during tokenization. "
             "This flag is useful when one is using uncased models (e.g. 'bert-base-uncased')",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        default=2,
        type=int,
        help="Batch size used during evaluation (per device). Defaults to 2.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate used for optimization. Defaults to 1e-5.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay factor. Defaults to 0 (no weight decay)."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Stability factor used in ADAM Optimizer, used to mitigate zero-division errors. "
             "Defaults to 1e-8."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=None,
        type=float,
        help="Maximum value of L2-norm of the gradients during optimization. Gradients "
             "with norm greater than this value will be clipped. Defaults to 1.0."
    )  # TODO!!!!!!!!!!!!!!!!

    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Saving interval in steps. Defaults to None.",
    )  # TODO!!!!!!!!!1
    parser.add_argument(
        "--save_total_limit",
        type=int,
        required=False,
        default=3,
        help="Maximum number of checkpoints that will be saved.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        required=False, # TODO!!!!
        choices=["f1_macro", "precision", "recall", "loss"],
        help="Metric used to compare model checkpoints. Checkpoints will be sorted according "
             "to the value of provided metric on the development sets."
    )

    # directories relevant for IO
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="The output directory where the model predictions and "
             "checkpoints will be written."
    )  # TODO

    # misc
    parser.add_argument(  # TODO - make use of this
        "--seed",
        type=int,
        default=42,
        help="Random seed. Will be used to perform dataset splitting, as well as "
             "random parameter initialization within the model. Defaults to 42."
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        required=False, # TODO
        help="Experiment name in MLFLow. If not provided, will not use MLFlow."
    )

    parser.add_argument(
        "--metrics_path",
        type=str,
        help="Path to which the metrics will be saved."
    )

    return parser
