import argparse
import logging

import pandas as pd

from pprint import pformat

from src.utils import setup_logging, dummy_preprocess_one

parser = argparse.ArgumentParser("Fine-tuning arguments parser")

parser.add_argument(
    "--dataframe_path", type=str, help="Filesystem path to a dataframe."
)

parser.add_argument(
    "--log_path",
    type=str,
    help="Filesystem path to a log file in which logs will be written.",
)


def main():
    args = parser.parse_args()
    setup_logging(args)

    logging.info(f"Dataset path = {args.dataframe_path}")

    df = pd.read_csv(args.dataframe_path)
    do_preprocess = dummy_preprocess_one()
    df = do_preprocess(df)

    logging.info(pformat(df.head()))


if __name__ == "__main__":
    main()
