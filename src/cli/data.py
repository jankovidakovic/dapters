from dataclasses import dataclass


@dataclass
class DataArguments:
    train_dataset_path: str
    eval_dataset_path: str
    preprocessing: str


def add_data_args(parser):
    group = parser.add_argument_group("data", "Arguments relevant for the data.")
    group.add_argument(
        "--train_dataset_path", type=str, help="Filesystem path to a training dataset."
    )

    group.add_argument(
        "--eval_dataset_path", type=str, help="Filesystem path to the evaluation dataset."
    )  # this is actually the "early stopping" split.


    group.add_argument(
        "--preprocessing",
        type=str,
        help="Name of the preprocessing method to use. See src/preprocess for available methods."

    )
