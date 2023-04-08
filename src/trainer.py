import logging
from typing import Callable, Any

import numpy as np
import torch.optim
from sklearn.metrics import multilabel_confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from torch.nn.functional import binary_cross_entropy_with_logits

from src.metrics import multilabel_classification_report
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
) -> list[dict[str, Any]]:
    # returns a list of classification metrics
    global_step = 0

    metrics = []

    for epoch in range(1, epochs + 1):
        for i, batch in (pbar := tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}")):
            # realno, za sad mi ne trebaju epohe
            # transfer tensors to gpu

            for key in batch:
                if type(batch[key]) == torch.Tensor:
                    batch[key] = batch[key].to("cuda")
            global_step += 1

            output = model(**batch)
            loss = loss_fn(
                input=output["logits"],
                target=batch["labels"],
                reduction="mean"
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if global_step % eval_steps == 0:
                model.eval()
                # evaluate
                with torch.no_grad():
                    current_metrics = validation(
                        model,
                        eval_dataloader,
                        label_names=label_names,
                        evaluation_threshold=evaluation_threshold,
                        loss_fn=loss_fn
                    )
                model.train()

                logger.info(current_metrics)  # why is this not logged then?
                metrics.append({
                    "global_step": global_step,
                    "metrics": current_metrics
                })

                # TODO - early stopping

                # now here we should log to MLFlow, right?

            if global_step % logging_steps == 0:
                # log loss
                pbar.set_description(desc=f"Epoch = {epoch}; Loss = {loss}", refresh=True)
                logger.info(f"Step = {global_step} : Loss = {loss}")
                #   this will fuck with tqdm tho, right?

            if global_step % save_steps == 0:
                save_checkpoint(
                    model=model,  # noqa
                    output_dir=output_dir,
                    global_step=global_step,
                    tokenizer=tokenizer
                )

            # I mean, technically this works, right?
    return metrics


def validation(
        model,
        eval_dataloader,
        label_names: list[str],
        evaluation_threshold: float = 0.75,
        loss_fn = binary_cross_entropy_with_logits
):
    confusion_matrix = np.zeros((len(label_names), 2, 2))
    total_loss = 0.0
    # we need some metrics here
    for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Validation"):
        for key in batch:
            if type(batch[key]) == torch.Tensor:
                batch[key] = batch[key].to("cuda")  # bruh?
        output = model(**batch)
        predictions = torch.sigmoid(output.logits)
        predictions = predictions > evaluation_threshold
        references = batch["labels"]

        total_loss += loss_fn(
            input=batch["labels"],
            target=output.logits,
            reduction="sum"  # not mean!
        ).item()

        # would also be cool to track loss, but I guess thats kinda useless?

        confusion_matrix += multilabel_confusion_matrix(
            y_true=references.detach().cpu().numpy(),
            y_pred=predictions.detach().cpu().numpy()  # noqa
        )

    metrics = multilabel_classification_report(
        confusion_matrix,
        label_names=label_names
    )  # TODO - this isnt really working, fix

    metrics["loss"] = total_loss / len(eval_dataloader.dataset)

    return metrics
