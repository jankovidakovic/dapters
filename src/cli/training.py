from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingArguments:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    max_grad_norm: Optional[float]
    dataloader_num_workers: int
    epochs: int
    logging_steps: int
    save_steps: int
    output_dir: str
    device: str
    use_tf32: bool
    early_stopping_patience: Optional[int]
    metric_for_best_model: Optional[str]
    greater_is_better: bool
    gradient_accumulation_steps: int
    evaluate_on_train: bool



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
             "with norm greater than this value will be clipped. Defaults to None"
    )

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
        "--logging_steps",
        type=int,
        default=100,
        help="Number of steps to perform between each loss logging."
    )

    group.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Period of model and tokenizer saving (in steps). "
             "Model and tokenizer are also saved at the end of each epoch."
             "If 'save_steps' is not specified, then no additional saving (besides the end of epoch) is done.",
    )

    group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of steps to wait before early stopping. Defaults to None."
    )

    group.add_argument(
        "--metric_for_best_model",
        type=str,
        default=None,
        help="Metric to use for early stopping. Defaults to None."
    )

    group.add_argument(
        "--greater_is_better",
        action="store_true",
        default=False,
        help="Whether the metric for best model is considered better when greater or lower. Defaults to True."
    )

    group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps. Defaults to 1."
    )

    group.add_argument(
        "--evaluate_on_train",
        action="store_true",
        default=False,
        help="Whether to evaluate on the training set (after every epoch). Defaults to False."
    )
