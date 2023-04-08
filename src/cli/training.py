from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class TrainingArguments:
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4
    epochs: int = 1
    eval_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./output/test"
    evaluation_threshold: float = 0.75
    device: str = "cuda"
    use_tf32: bool = True


def add_training_args(parser: ArgumentParser):
    group = parser.add_argument_group("training", "Training arguments")

    group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device. Defaults to 4."
    )

    group.add_argument(
        "--per_device_eval_batch_size",
        default=2,
        type=int,
        help="Batch size used during evaluation (per device). Defaults to 2.",
    )

    group.add_argument(
        "--max_grad_norm",
        default=None,
        type=float,
        help="Maximum value of L2-norm of the gradients during optimization. Gradients "
             "with norm greater than this value will be clipped. Defaults to 1.0."
    )  # TODO!!!!!!!!!!!!!!!!

    group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers. Defaults to 4."
    )

    group.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs. Defaults to 1."
    )

    group.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of gradient steps to perform between each evaluation."
    )

    group.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of steps to perform between each loss logging."
    )

    group.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Saving interval in steps. Defaults to None.",
    )

    group.add_argument(
        "--evaluation_threshold",
        type=float,
        default=0.75,
        help="Threshold for classification evaluation. Defaults to 0.75."
    )
