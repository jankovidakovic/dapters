# extract optimizer arguments from cli into here
from argparse import ArgumentParser
from dataclasses import dataclass, field

from transformers import SchedulerType


@dataclass
class OptimizerArguments:
    learning_rate: float
    adam_epsilon: float
    weight_decay: float
    scheduler_type: str | SchedulerType
    warmup_percentage: float

    def __post_init__(self):
        self.scheduler_type = SchedulerType(self.scheduler_type)


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
    optimizer_group.add_argument(
        "--scheduler_type",
        default="constant",
        choices=["linear", "constant", "cosine", "polynomial", "constant_with_warmup"],
        type=str,
        help="The type of scheduler to use. Defaults to constant.",
    )

    optimizer_group.add_argument(
        "--warmup_percentage",
        type=float,
        default=0,
        help="Percentage of training for which the linear warmup will be used. Defaults to 0, which means no warmup."
    )
