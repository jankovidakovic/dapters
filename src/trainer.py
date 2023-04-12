import logging
from pprint import pformat
from typing import Callable, Optional

import mlflow
import numpy as np
import torch.optim
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from torch.nn.functional import binary_cross_entropy_with_logits

from src.utils import save_checkpoint, is_improved

logger = logging.getLogger(__name__)


def train(
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        epochs: int,
        eval_steps: int,  # does it even make sense to decouple this?
        save_steps: int,
        output_dir: str,
        logging_steps: int,
        label_names: list[str],
        loss_fn: Callable = binary_cross_entropy_with_logits,
        evaluation_threshold: float = 0.75,
        max_grad_norm: float = 1.0,
        early_stopping_patience: Optional[int] = None,
        metric_for_best_model: str = "macro-f1",
        greater_is_better: bool = True,
        gradient_accumulation_steps: int = 1
):
    global_step = 0
    early_stopping_step: Optional[int]
    best_metric_value: Optional[float]

    if early_stopping_patience:
        logger.warning(f"Early stopping is enabled with patience of {early_stopping_patience}."
                       f"Metric for best model is {metric_for_best_model}, for which "
                       f"{'higher' if greater_is_better else 'lower'} values are better.")
        best_metric_value = np.inf
        if greater_is_better:
            best_metric_value *= -1
        early_stopping_step = 0

    for epoch in range(1, epochs + 1):
        epoch_step = 0
        for i, batch in (pbar := tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}")):
            epoch_step += 1

            for key in batch:
                if type(batch[key]) == torch.Tensor:
                    batch[key] = batch[key].to(model.device)
            global_step += 1

            output = model(**batch)
            loss = loss_fn(
                input=output["logits"],
                target=batch["labels"],
                reduction="mean"
            )
            loss.backward()

            if (epoch_step % gradient_accumulation_steps == 0
                or epoch_step == len(train_dataloader)
            ):
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % eval_steps == 0:
                # evaluate
                metrics = validation(
                    model,
                    eval_dataloader,
                    label_names=label_names,
                    global_step=global_step,
                    evaluation_threshold=evaluation_threshold,
                    loss_fn=loss_fn
                )

                logger.info(f"[GLOBAL_STEP = {global_step}] {pformat(metrics)}")
                mlflow.log_metrics(
                    metrics=metrics,
                    step=global_step
                )

                # TODO - early stopping
                # now we implement early stopping

                # step 1 -> extract metric
                current_metric_value = metrics[metric_for_best_model]
                if not is_improved(
                    current_metric_value,
                    best_metric_value,  # noqa
                    greater_is_better
                ):
                    early_stopping_step += 1  # noqa
                    if early_stopping_step == early_stopping_patience:
                        # early stopping
                        logger.warning(f"Early stopping patience has reached the critical threshold of "
                                       f"{early_stopping_patience}. Stopping the run.")
                        return
                else:
                    logger.warning(f"""Resetting early stopping patience based on {metric_for_best_model}.
                                   Current value: {current_metric_value}
                                   Best_value: {best_metric_value}""")
                    early_stopping_step = 0
                    best_metric_value = current_metric_value


            if global_step % logging_steps == 0:
                # log loss
                pbar.set_description(desc=f"Epoch = {epoch}; Loss = {loss}", refresh=True)
                mlflow.log_metric(key="train_loss", value=loss.item(), step=global_step)
                #   this will fuck with tqdm tho, right?

            if global_step % save_steps == 0:
                save_checkpoint(
                    model=model,  # noqa
                    output_dir=output_dir,
                    global_step=global_step,
                    tokenizer=tokenizer
                )

            # I mean, technically this works, right?


def validation(
        model,
        eval_dataloader,
        label_names: list[str],
        global_step: int,
        evaluation_threshold: float = 0.75,
        loss_fn = binary_cross_entropy_with_logits,
):
    # we need some metrics here
    model.eval()  # this would be prettier as a context manager, but whatever

    # initialize tensors for predictions and references
    eval_size = len(eval_dataloader.dataset)
    logits_all = torch.empty(eval_size, 33, device="cpu", dtype=torch.float32)
    references_all = torch.empty(eval_size, 33, device="cpu", dtype=torch.float32)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Validation"):
            for key in batch:
                if type(batch[key]) == torch.Tensor:
                    batch[key] = batch[key].to(model.device)  # bruh?
            output = model(**batch)
            references = batch["labels"]

            batch_slice = slice(i * eval_dataloader.batch_size, (i + 1) * eval_dataloader.batch_size)
            logits_all[batch_slice] = output["logits"].detach().cpu()
            references_all[batch_slice] = references.detach().cpu()

    # compute loss

    metrics = {
        "eval_loss": loss_fn(
            input=logits_all,
            target=references_all,
            reduction="mean"
        ).item()
    }

    predictions_all = (torch.sigmoid(logits_all) > evaluation_threshold).int()  # noqa

    averages = ["macro", "micro", "weighted"]

    for average in averages:
        prf = precision_recall_fscore_support(
            y_true=references_all,
            y_pred=predictions_all,
            average=average,
            zero_division=0
        )[:-1]

        for name, value in zip(["precision", "recall", "f1"], prf):
            metrics[f"{average}-{name}"] = value

    model.train()

    return metrics
