import logging
from pprint import pformat
from typing import Callable, Optional

import mlflow
import numpy as np
import torch.optim
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding

from torch.nn.functional import binary_cross_entropy_with_logits
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
from transformers.utils import ModelOutput

from src.utils import save_checkpoint, is_improved, set_device

logger = logging.getLogger(__name__)


# well, we need different trainers for pretraining and finetuning
# alternatively, we can pass in the evaluation function as an argument
#   -> sounds better


def fine_tuning_loss(
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    """ If loss_fn takes in additional arguments, such arguments
    should be partially applied to loss_fn before passing it to
    this function.

    :param loss_fn:
    :return:
    """
    def loss(
            batch: BatchEncoding,
            model_output: SequenceClassifierOutput
    ):
        return loss_fn(
            model_output.logits,
            batch["labels"],
        )

    return loss


def pretraining_loss():
    def loss(
            batch: BatchEncoding,
            model_output: MaskedLMOutput
    ):
        return model_output.loss

    return loss


# TODO - there is code duplication of inner functions, think about how to refactor


def train(
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        epochs: int,
        eval_steps: int,  # does it even make sense to decouple this?
        save_steps: int,
        output_dir: str,
        logging_steps: int,
        do_evaluate: Callable[[nn.Module, DataLoader], dict[str, float]],
        get_loss: Callable[[BatchEncoding, ModelOutput], torch.Tensor],
        max_grad_norm: float = 1.0,
        early_stopping_patience: Optional[int] = None,
        metric_for_best_model: str = "macro-f1",
        greater_is_better: bool = True,
        gradient_accumulation_steps: int = 1,
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
            global_step += 1

            set_device(batch, model.device)

            output = model(**batch)
            loss = get_loss(batch=batch, model_output=output)  # noqa
            loss.backward()

            if (epoch_step % gradient_accumulation_steps == 0
                or epoch_step == len(train_dataloader)
            ):
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # updates the learning rate
                optimizer.zero_grad()
                pbar.set_description(
                    f"Epoch = {epoch} (LR = {scheduler.get_last_lr()[-1]:.8f}; loss = {loss.item():.4f})"
                )

            if global_step % eval_steps == 0:
                # evaluate
                metrics = do_evaluate(model, eval_dataloader)

                logger.info(f"[GLOBAL_STEP = {global_step}] {pformat(metrics)}")
                mlflow.log_metrics(
                    metrics=metrics,
                    step=global_step
                )

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
                mlflow.log_metric(key="train_loss", value=loss.item(), step=global_step)
                # TODO - log average loss instead of current loss

            if global_step % save_steps == 0:
                save_checkpoint(
                    model=model,  # noqa
                    output_dir=output_dir,
                    global_step=global_step,
                    tokenizer=tokenizer
                )

            # I mean, technically this works, right?


def evaluate_finetuning(
        evaluation_threshold: float = 0.75,
        loss_fn = binary_cross_entropy_with_logits
) -> Callable[[nn.Module, DataLoader], dict[str, float]]:
    def evaluate(
        model: nn.Module,
        eval_dataloader: DataLoader,
    ) -> dict[str, float]:

        # we need some metrics here
        model.eval()  # this would be prettier as a context manager, but whatever

        # initialize tensors for predictions and references
        eval_size = len(eval_dataloader.dataset)  # noqa
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

    return evaluate


def evaluate_pretraining():
    # metrics = ["accuracy", "eval_loss", "perplexity"]
    def do_evaluate(
            model: nn.Module,
            eval_dataloader: DataLoader,
    ):
        model.eval()  # this would be prettier as a context manager, but whatever

        # initialize tensors for predictions and references
        eval_size = len(eval_dataloader.dataset)  # noqa
        # logits_all = torch.empty(eval_size, 33, device="cpu", dtype=torch.float32)
        # references_all = torch.empty(eval_size, 33, device="cpu", dtype=torch.float32)

        losses = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Validation"):
                set_device(batch, model.device)
                output: MaskedLMOutput = model(**batch)
                # references = batch["labels"]
                # batch_slice = slice(i * eval_dataloader.batch_size, (i + 1) * eval_dataloader.batch_size)
                # logits_all[batch_slice] = output.logits.detach().cpu()
                # references_all[batch_slice] = references.detach().cpu()
                losses.append(output.loss.item())

        # compute loss

        metrics = {
            "eval_loss": np.mean(losses)
        }

        model.train()
        return metrics


    return do_evaluate