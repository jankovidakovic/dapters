from dataclasses import dataclass


@dataclass
class DataArguments:
    train_dataset_path: str
    eval_dataset_path: str
    labels_path: str


def add_data_args(parser):
    group = parser.add_argument_group("data", "Arguments relevant for the data.")
    group.add_argument(
        "--train_dataset_path", type=str, help="Filesystem path to a training dataset."
    )

    group.add_argument(
        "--eval_dataset_path", type=str, help="Filesystem path to the evaluation dataset."
    )  # this is actually the "early stopping" split.

    group.add_argument(
        "--labels_path",
        type=str,
        help="Path to a JSON file with the layout {'labels': ['LABEL_0', 'LABEL_1', ...]}"
    )
