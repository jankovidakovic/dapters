import argparse

from src.cli.data import add_data_args
from src.cli.model import add_model_args
from src.cli.optimizer import add_optimizer_args
from src.cli.tokenizer import add_tokenizer_args


def get_parser():
    parser = argparse.ArgumentParser("Fine-tuning arguments parser")

    parser.add_argument(
        "--log_path",
        type=str,
        help="Filesystem path to a log file in which logs will be written.",
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
    )
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
    )

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

    parser.add_argument(
        "--use_tf32",
        action="store_true",
        help="If set, will use TF32 precision to compute matrix multiplications and convolutions."
    )

    add_model_args(parser)
    add_tokenizer_args(parser)
    add_optimizer_args(parser)
    add_data_args(parser)

    return parser
