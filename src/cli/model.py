from argparse import ArgumentParser
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str
    problem_type: str
    cache_dir: str


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

    group.add_argument(
        "--problem_type",
        type=str,
        choices=["multi-label"],  # TODO - add masked language modelling
        default="multi-label",
        help="Problem type."
    )

