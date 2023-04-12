import logging
from pprint import pformat
from typing import Callable

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

from src.utils import save_checkpoint

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
        max_grad_norm: float = 1.0
):
    # returns a list of classification metrics
    global_step = 0

    for epoch in range(1, epochs + 1):
        for i, batch in (pbar := tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}")):
            # realno, za sad mi ne trebaju epohe
            # transfer tensors to gpu

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
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # TODO - gradient accumulation

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
    confusion_matrix = np.zeros((len(label_names), 2, 2))
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

    predictions_all = torch.tensor(torch.sigmoid(logits_all) > evaluation_threshold).int()

    averages = ["macro", "micro", "weighted"]

    for average in averages:
        prf = precision_recall_fscore_support(
            y_true=references_all,
            y_pred=predictions_all,
            average=average,
            zero_division=0
        )[:-1]

        for name, value in zip(["precision", "recall", "f1-score"], prf):
            metrics[f"{average}-{name}"] = value

    model.train()

    return metrics
