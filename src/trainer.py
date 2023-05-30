import gc
import logging
from pprint import pformat
from typing import Callable, Optional, Union, Tuple

import numpy as np
import torch.optim
from math import ceil
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollator

from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
from transformers.utils import ModelOutput

from src.utils import save_checkpoint, is_improved, set_device, save_transformer_model, get_cls_token

logger = logging.getLogger(__name__)


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
        args,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        epochs: int,
        get_loss: Callable[[BatchEncoding, ModelOutput], torch.Tensor],
        max_grad_norm: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
        do_evaluate: Optional[Callable[[nn.Module, DataLoader], dict[str, float]]] = None,
        use_mlflow: bool = False,
        evaluate_on_train: bool = False,
        collate_fn: Optional[DataCollator] = None,
        use_ray_tune: bool = False,
        model_saving_callback: Callable = save_transformer_model,
        dataloader_num_workers: int = 8,
):
    global_step = 0
    early_stopping_step: Optional[int]
    best_metric_value: Optional[float]

    if use_early_stopping := hasattr(args, "early_stopping"):
        logger.warning(f"Early stopping is enabled with patience of {args.early_stopping.patience}."
                       f"Metric for best model is {args.early_stopping.metric_for_best_model}, for which "
                       f"{'higher' if args.early_stopping.greater_is_better else 'lower'} values are better.")
        best_metric_value = np.inf
        if args.early_stopping.greater_is_better:
            best_metric_value *= -1
        early_stopping_step = 0

    if max_grad_norm:
        logger.warning(f"Gradient clipping is enabled with max norm of {max_grad_norm}.")

    if gradient_accumulation_steps > 1:
        logger.warning(f"Gradient accumulation is enabled with {gradient_accumulation_steps} steps.")

    if use_mlflow:
        import mlflow
        logger.warning(f"MLFlow is enabled. Logging to {mlflow.get_tracking_uri()}.")

    if use_ray_tune:
        from ray.air import session

    # setup train dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn,
        # hardcoded for now
    )

    if evaluate_on_train:
        train_eval_dataloader = DataLoader(
            train_dataset,
            batch_size=per_device_eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=dataloader_num_workers,
            collate_fn=collate_fn
        )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn
    )

    epoch_steps = ceil(len(train_dataset) / per_device_train_batch_size)
    if len(train_dataloader) != epoch_steps:
        logger.warning(f"Epoch steps is {epoch_steps}, but dataloader has {len(train_dataloader)} batches.")
        raise RuntimeError("ree")

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
                # clip gradients if enabled
                if max_grad_norm:
                    clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # updates the learning rate
                optimizer.zero_grad()
                pbar.set_description(
                    f"Epoch = {epoch} (LR = {scheduler.get_last_lr()[-1]:.8f}; loss = {loss.item():.4f})"
                )

        # save checkpoint at the end of the epoch
        logger.warning(f"Saving checkpoint at the end of epoch {epoch}...")
        save_checkpoint(
            model=model,  # noqa
            checkpoint_name=f"{epoch}-ckpt",
            use_mlflow=use_mlflow,
            model_saving_callback=model_saving_callback
        )

        if evaluate_on_train:
            logger.warning(f"Evaluating on training set after epoch {epoch}...")
            metrics = do_evaluate(model, train_eval_dataloader, prefix="train")
            # yea but we would want to do this with the full-blown batch size
            logger.warning(f"[EPOCH = {epoch}; GLOBAL_STEP = {global_step}] {pformat(metrics)}")

            if use_mlflow:
                mlflow.log_metrics(
                    metrics=metrics,
                    step=global_step
                )

        if eval_dataloader:
            logger.warning(F"Evaluating...")
            metrics = do_evaluate(model, eval_dataloader, prefix="eval")
            logger.warning(f"[EPOCH = {epoch}; GLOBAL_STEP = {global_step}] {pformat(metrics)}")

            if use_mlflow:
                mlflow.log_metrics(
                    metrics=metrics,
                    step=global_step
                )

            if use_ray_tune:
                session.report(
                    {"epoch": epoch, **metrics}
                )  # this KILLS the run if the metrics are bad

            if use_early_stopping and epoch >= args.early_stopping.start:
                current_metric_value = metrics[args.early_stopping.metric_for_best_model]
                if not is_improved(
                        current_metric_value,
                        best_metric_value,  # noqa
                        args.early_stopping.greater_is_better
                ):
                    early_stopping_step += 1  # noqa
                    if early_stopping_step == args.early_stopping.patience:
                        # early stopping
                        logger.warning(f"Early stopping patience has reached the critical threshold of "
                                       f"{args.early_stopping.patience}. Stopping the run.")
                        return
                else:
                    logger.warning(
                        f"""Resetting early stopping patience based on {args.early_stopping.metric_for_best_model}.
                                   Current value: {current_metric_value}
                                   Best_value: {best_metric_value}""")
                    early_stopping_step = 0
                    best_metric_value = current_metric_value
                    # at this point, we save as "best_checkpoint"
                    save_checkpoint(
                        model=model,  # noqa
                        checkpoint_name="best_checkpoint",
                        use_mlflow=use_mlflow,
                        model_saving_callback=model_saving_callback,
                    )


def eval_loss_only(
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[nn.Module, DataLoader], dict[str, float]]:
    def evaluate(
            model: nn.Module,
            eval_dataloader: DataLoader
    ) -> dict[str, float]:
        model.eval()

        # initialize tensors for predictions and references
        eval_size = len(eval_dataloader.dataset)  # noqa
        logits_all = torch.empty(eval_size, 33, device="cpu", dtype=torch.float32)
        references_all = torch.empty(eval_size, 33, device="cpu", dtype=torch.float32)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Validation"):
                set_device(batch, model.device)
                output = model(**batch)
                references = batch["labels"]

                batch_slice = slice(i * eval_dataloader.batch_size, (i + 1) * eval_dataloader.batch_size)
                logits_all[batch_slice] = output["logits"].detach().cpu()
                references_all[batch_slice] = references.detach().cpu()

        # compute loss

        metrics = {
            "eval_loss": loss_fn(logits_all, references_all).item()
        }

        model.train()

        return metrics

    return evaluate


@torch.no_grad()
def do_predict(
        model: nn.Module,
        dataloader: DataLoader,
        output_hidden_states: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """ Runs inference using the given dataloader.
    Model outputs are transformed to probabilities using sigmoid function.

    :param model:
    :param dataloader:
    :return: tensor of shape (num_samples, num_classes)
    """

    data_len = len(dataloader.dataset)  # noqa
    num_labels = model.config.num_labels
    predictions = torch.empty(data_len, num_labels, device="cpu", dtype=torch.float32)
    references = torch.empty(data_len, num_labels, device="cpu", dtype=torch.float32)

    if output_hidden_states:
        hidden_states = torch.empty(data_len, model.config.hidden_size, device="cpu", dtype=torch.float32)

    model.eval()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Prediction loop"):
        set_device(batch, model.device)
        output = model(**batch, output_hidden_states=output_hidden_states)
        print(output.keys())
        batch_slice = slice(i * dataloader.batch_size, (i + 1) * dataloader.batch_size)

        if output_hidden_states:
            hidden_states[batch_slice, :] = get_cls_token(output.hidden_states[-1]).detach().cpu().numpy()

        predictions[batch_slice] = output["logits"].detach().cpu()
        references[batch_slice] = batch["labels"].detach().cpu()

    model.train()

    if output_hidden_states:
        return predictions, references, hidden_states
    else:
        return predictions, references


def compute_metrics(
        predictions,
        references,
        prefix: str,
        loss_fn=binary_cross_entropy_with_logits,
        evaluation_threshold: float = 0.75
):
    metrics = {
        f"{prefix}_loss": loss_fn(
            input=predictions,
            target=references,
            reduction="mean"
        ).item()
    }

    predictions = (torch.sigmoid(predictions) > evaluation_threshold).int()  # noqa

    averages = ["macro", "micro", "weighted"]

    for average in averages:
        prf = precision_recall_fscore_support(
            y_true=references,
            y_pred=predictions,
            average=average,
            zero_division=0
        )[:-1]

        for name, value in zip(["precision", "recall", "f1"], prf):
            metrics[f"{prefix}_{average}-{name}"] = value

    return metrics


def evaluate_finetuning(
        evaluation_threshold: float = 0.75,
        loss_fn=binary_cross_entropy_with_logits,
) -> Callable[[nn.Module, DataLoader, str], dict[str, float]]:
    def evaluate(
            model: nn.Module,
            eval_dataloader: DataLoader,
            prefix: str = "eval"
    ) -> dict[str, float]:
        model.eval()
        predictions, references = do_predict(model, eval_dataloader)
        model.train()

        return compute_metrics(predictions, references, prefix, loss_fn, evaluation_threshold)

    return evaluate


def evaluate_pretraining():
    def do_evaluate(
            model: nn.Module,
            eval_dataloader: DataLoader,
            prefix: str = "eval"
    ):
        model.eval()

        # initialize tensors for predictions and references
        eval_size = len(eval_dataloader.dataset)  # noqa

        total_loss = 0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Validation"):
                set_device(batch, model.device)
                output: MaskedLMOutput = model(**batch)

                batch_loss = cross_entropy(
                    input=output.logits.detach().cpu().view(-1, model.config.vocab_size),
                    target=batch["labels"].detach().cpu().view(-1),
                    reduction="mean"
                ).item()
                current_batch_size = batch["input_ids"].shape[0]
                total_loss += batch_loss * current_batch_size

        metrics = {
            f"{prefix}_loss": total_loss / len(eval_dataloader.dataset)
        }

        model.train()
        return metrics

    return do_evaluate
