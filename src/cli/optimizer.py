# extract optimizer arguments from cli into here
from argparse import ArgumentParser
from dataclasses import dataclass, field


@dataclass
class OptimizerArguments:
    learning_rate: float = field(default=2e-5)
    adam_epsilon: float = field(default=1e-8)
    weight_decay: float = field(default=0.0)


def add_optimizer_args(parser: ArgumentParser):
    # add optimizer argument group
    optimizer_group = parser.add_argument_group(
        "optimizer",
        "Arguments related to the optimizer."
    )

    # add optimizer arguments
    optimizer_group.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate used for optimization. Defaults to 2e-5.",
    )
    optimizer_group.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Stability factor used in ADAM Optimizer, used to mitigate zero-division errors. "
             "Defaults to 1e-8.",
    )
    optimizer_group.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay factor. Defaults to 0 (no weight decay)."
    )
