from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class ConfigurationArguments:
    device: str
    use_tf32: bool
    random_seed: int
    mlflow_experiment: str
    log_path: str
    metrics_path: str
    output_dir: str



def add_configuration_args(parser: ArgumentParser):
    group = parser.add_argument_group("configuration", "Configuration arguments")

    group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training. Defaults to cuda."
    )

    group.add_argument(
        "--use_tf32",
        action="store_true",
        default=True,
        help="Whether to use TensorFloat32. Defaults to True."
    )

    group.add_argument(
        "--random_seed",
        type=int,
        default=192837465,
        help="Random seed. Defaults to 192837465."
    )  # TODO

    group.add_argument(
        "--mlflow_experiment",
        type=str,
        default=None,
        help="Name of the MLFlow experiment. Defaults to None."
    )  # TODO

    group.add_argument(
        "--log_path",
        type=str,
        default="./run.log",
        help="Path to the log file. Defaults to ./run.log."
    )

    group.add_argument(
        "--metrics_path",
        type=str,
        default="./metrics.json",
        help="Path to where the metrics will be saved. Defaults to ./metrics.json."
    )

    # directories relevant for IO
    group.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./output/test",
        help="The output directory where the model predictions and "
             "checkpoints will be written."
    )

    return parser