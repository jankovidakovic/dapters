from dataclasses import dataclass
from typing import Optional


@dataclass
class DataArguments:
    train_dataset_path: str
    eval_dataset_path: Optional[str]
    message_column: str


def add_data_args(parser):
    group = parser.add_argument_group("data", "Arguments relevant for the data.")
    group.add_argument(
        "--train_dataset_path", type=str, help="Filesystem path to a training dataset."
    )

    group.add_argument(
        "--eval_dataset_path",
        type=str,
        help="Filesystem path to the evaluation dataset.",
        default=None
    )  # this is actually the "early stopping" split.

    group.add_argument(
        "--message_column",
        type=str,
        default="preprocessed",
        help="Name of the column containing the message to be classified."
    )
