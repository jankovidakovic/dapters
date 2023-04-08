import argparse
from dataclasses import dataclass

from src.cli.config import add_configuration_args, ConfigurationArguments
from src.cli.data import add_data_args, DataArguments
from src.cli.model import add_model_args, ModelArguments
from src.cli.optimizer import add_optimizer_args, OptimizerArguments
from src.cli.tokenizer import add_tokenizer_args, TokenizerArguments
from src.cli.training import add_training_args, TrainingArguments


@dataclass
class FineTuningArguments(
    ModelArguments,
    TokenizerArguments,
    OptimizerArguments,
    DataArguments,
    TrainingArguments,
    ConfigurationArguments
):
    pass

def get_parser():
    parser = argparse.ArgumentParser("Fine-tuning arguments parser")

    # parser.add_argument(
        # "--save_total_limit",
        # # type=int,
        # required=False,
        # default=3,
        # help="Maximum number of checkpoints that will be saved.",
    # )
    # parser.add_argument(
        # "--metric_for_best_model",
        # type=str,
        # required=False, # TODO!!!!
        # choices=["f1_macro", "precision", "recall", "loss"],
        # help="Metric used to compare model checkpoints. Checkpoints will be sorted according "
             # "to the value of provided metric on the development sets."
    # )

    add_model_args(parser)
    add_tokenizer_args(parser)
    add_optimizer_args(parser)
    add_data_args(parser)
    add_training_args(parser)
    add_configuration_args(parser)

    return parser

def parse_args():
    args = get_parser().parse_args()
    return FineTuningArguments(**vars(args))
