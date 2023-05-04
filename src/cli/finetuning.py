import argparse

from src.cli.common import add_configuration_args, add_tokenizer_args, add_model_args, add_optimizer_args, \
    add_data_args, add_training_args


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
