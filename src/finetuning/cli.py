import argparse

from src.cli.config import add_configuration_args
from src.cli.data import add_data_args
from src.cli.model import add_model_args
from src.cli.optimizer import add_optimizer_args
from src.cli.tokenizer import add_tokenizer_args
from src.cli.training import add_training_args


def get_parser():
    parser = argparse.ArgumentParser("Fine-tuning arguments parser")

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="If set, input text will be lowercased during tokenization. "
             "This flag is useful when one is using uncased models (e.g. 'bert-base-uncased')",
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

    add_model_args(parser)
    add_tokenizer_args(parser)
    add_optimizer_args(parser)
    add_data_args(parser)
    add_training_args(parser)
    add_configuration_args(parser)

    return parser
