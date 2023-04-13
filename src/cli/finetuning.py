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
    labels_path: str
    evaluation_threshold: float

def get_parser():
    parser = argparse.ArgumentParser("Fine-tuning arguments parser")

    add_model_args(parser)
    add_tokenizer_args(parser)
    add_optimizer_args(parser)
    add_data_args(parser)
    add_training_args(parser)
    add_configuration_args(parser)

    parser.add_argument(
        "--labels_path",
        type=str,
        help="Filesystem path to the JSON file containing a list of labels."
    )

    parser.add_argument(
        "--evaluation_threshold",
        type=float,
        default=0.75,
        help="Threshold for classification evaluation. Defaults to 0.75."
    )

    return parser

def parse_args():
    args = get_parser().parse_args()
    return FineTuningArguments(**vars(args))
