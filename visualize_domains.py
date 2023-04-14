import logging
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DefaultDataCollator, PreTrainedModel, PreTrainedTokenizer
from transformers.utils import PaddingStrategy

from src.preprocess.steps import keep, deduplication, sample_by, to_hf_dataset, hf_map, convert_to_torch, \
    sequence_columns

from src.utils import pipeline, get_tokenization_fn, set_device

logger = logging.getLogger(__name__)



@dataclass
class CLI:
    save_path: str
    message_column: str
    dataset_paths: list[str]
    sample_size: int
    cache_dir: str
    model_name_or_path: str
    tokenizer_path: str
    batch_size: int
    pca_on_first_only: bool


def get_args() -> CLI:
    parser = ArgumentParser(
        "Visualization of hidden representations of transformers."
    )
    parser.add_argument(
        "--pca_on_first_only",
        action="store_true",
        help="If set, PCA will be performed only on the first dataset."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="pca.png"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data2/jvidakovic/.cache/huggingface",

    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size."
    )
    parser.add_argument(
        "--message_column",
        type=str,
        default="message",
        help="Name of the column containing the message."
    )
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to datasets."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Sample size."
    )

    args = parser.parse_args()
    return CLI(**vars(args))


def main():
    # load model
    args: CLI = get_args()

    model: PreTrainedModel = AutoModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir
    )

    model = torch.compile(model).to("cuda")  # noqa
    model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        cache_dir=args.cache_dir,
        do_lower_case=True,
    )

    do_tokenize = get_tokenization_fn(
        tokenizer,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=True,
        max_length=64,
        # return_special_tokens_mask=True
    )

    # create the dataset processing pipeline
    do_process = pipeline(
        pd.read_csv,
        keep([args.message_column]),
        deduplication(args.message_column),
        sample_by(args.sample_size),
        to_hf_dataset,
        hf_map(do_tokenize, batched=True),
        convert_to_torch(columns=sequence_columns)
    )

    all_embeddings = []

    for i, dataset_path in enumerate(args.dataset_paths):
        dataset = do_process(dataset_path)
        logger.warning(f"Dataset {i} finished processing.")
        sequence_embeddings = np.zeros((len(dataset), model.config.hidden_size))

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=DefaultDataCollator(return_tensors="pt")
        )

        with torch.no_grad():
            for j, batch in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    desc="Inference"
            ):
                set_device(batch, model.device)
                model_output = model(**batch)
                s = slice(j * args.batch_size,(j+1)*args.batch_size)
                sequence_embeddings[s, :] = model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()

        all_embeddings.append(sequence_embeddings)

    # now do the PCA
    pca = PCA(n_components=2)
    if args.pca_on_first_only:
        logger.warning(f"Fitting PCA on first dataset only.")
        pca.fit(all_embeddings[0])
    else:
        logger.warning(f"Fitting PCA on all data.")
        pca.fit(np.concatenate(all_embeddings, axis=0))

    pca_representations = [
        pca.transform(embeddings) for embeddings in all_embeddings
    ]

    # plot
    for i, pca_repr in enumerate(pca_representations):
        plt.scatter(
            pca_repr[:, 0],
            pca_repr[:, 1],
            marker=".",
            alpha=0.5,
            label=f"Dataset {i}"
        )

    plt.grid()
    plt.xlabel(f"1st principal component ({pca.explained_variance_ratio_[0] * 100:.2f}% $\sigma^2$)")
    plt.ylabel(f"2nd principal component ({pca.explained_variance_ratio_[1] * 100:.2f}% $\sigma^2$)")

    plt.legend()
    plt.title("PCA-transformed hidden representations of [CLS] tokens.")
    plt.savefig(args.save_path)


if __name__ == "__main__":
    main()