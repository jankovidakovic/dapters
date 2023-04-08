from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class TokenizerArguments:
    pretrained_model_name_or_path: str
    padding: str
    truncation: bool
    max_length: int
    do_lower_case: bool


def add_tokenizer_args(parser: ArgumentParser):
    group = parser.add_argument_group(
        "Tokenizer arguments",
        "Tokenizer arguments"
    )

    group.add_argument(
        "--padding",
        type=str,
        choices=["longest", "max_length", "do_not_pad"],
        default="max_length",
        help="Padding strategy when tokenizing. Defaults to 'max_length'."
    )

    group.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Model max length. Defaults to 64."
    )

    group.add_argument(
        "--truncation",
        action="store_true",
        default=True,
        help="If set, will truncate sequences for which length exceeds --max_length."
    )

    group.add_argument(
        "--do_lower_case",
        action="store_true",
        help="If set, input text will be lowercased during tokenization. "
                "This flag is useful when one is using uncased models (e.g. 'bert-base-uncased')",
    )

