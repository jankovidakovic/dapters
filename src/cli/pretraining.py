import argparse
from dataclasses import dataclass

from src.cli.common import add_configuration_args, add_tokenizer_args, add_model_args, add_optimizer_args, \
    add_data_args, add_training_args


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
