import argparse
from dataclasses import dataclass

from src.cli.config import add_configuration_args, ConfigurationArguments
from src.cli.data import add_data_args, DataArguments
from src.cli.model import add_model_args, ModelArguments
from src.cli.optimizer import add_optimizer_args, OptimizerArguments
from src.cli.tokenizer import add_tokenizer_args, TokenizerArguments
from src.cli.training import add_training_args, TrainingArguments


@dataclass
class PreTrainingArguments(
    ModelArguments,
    TokenizerArguments,
    OptimizerArguments,
    DataArguments,
    TrainingArguments,
    ConfigurationArguments
):  # theres actually no difference, right? at least for now
    mlm_probability: float

def get_parser():
    parser = argparse.ArgumentParser("Pretraining argument parser.")

    add_model_args(parser)
    add_tokenizer_args(parser)
    add_optimizer_args(parser)
    add_data_args(parser)
    add_training_args(parser)
    add_configuration_args(parser)

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masking each token for the masked language model."
    )

    return parser

def parse_args():
    args = get_parser().parse_args()
    return PreTrainingArguments(**vars(args))
