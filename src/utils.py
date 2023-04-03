from typing import Callable
import pandas as pd
import logging
import os


def pipeline(*fs) -> Callable:
    """Performs a forward composition of given functions.

    For example, if some three functions are
        A : x -> y
        B : y -> z
        C : z -> w

    then pipeline([A, B, C]) will return a function F : x -> w

    :param fs:  Functions to compose.
    :return:  A callable object that when called, calls the given functions
             in a sequential order and returns the result.
    """

    def pipe(*args, **kwargs):
        output = fs[0](*args, **kwargs)
        for f in fs[1:]:
            output = f(output)
        return output

    return pipe


def sample_by(column: str, sample_size: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def apply(df: pd.DataFrame):
        return df.groupby(column).sample(sample_size)

    return apply


def dummy_preprocess_one() -> Callable[[pd.DataFrame], pd.DataFrame]:
    return pipeline(
        lambda df: df.drop(
            columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags"]
        ),
        lambda df: df.drop_duplicates(subset="message"),
        sample_by("cluster_id", 1),
        lambda df: df.drop(columns=["cluster_id"]),
    )


def make_logfile_name(args):
    return args.log_path


def setup_logging(args):
    # logging
    dirname = os.path.dirname(os.path.abspath(args.log_path))
    os.makedirs(dirname, exist_ok=True)
    log_filename = make_logfile_name(args)

    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # log info only on main process
        level=logging.INFO,  # TODO - info only if verbose?
        handlers=[
            logging.FileHandler(filename=log_filename, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging to file: {os.path.abspath(log_filename)}")
