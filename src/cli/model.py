from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str
    num_labels: int
    problem_type: str


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
        "--num_labels",
        type=int,
        default=None,
        required=True,
        help="Number of unique labels in dataset. If not provided, will "
             "be calculated as the number of unique values in labels column "
             "in the training dataset. Name of the column containig labels can "
             "be set using the '--label_column' option.",
    )

    group.add_argument(
        "--problem_type",
        type=str,
        choices=["multi-label"],  # TODO - add masked language modelling
        help="Problem type."
    )
