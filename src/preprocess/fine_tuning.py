from src.utils import pipeline, sample_by


dummy = pipeline(
    lambda df: df.drop(
        columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags"]
    ),
    lambda df: df.drop_duplicates(subset="message"),
    sample_by("cluster_id", 1),
    lambda df: df.drop(columns=["cluster_id"]),
)


fine_tuning_dev = pipeline(
    lambda df: df.drop(
        columns=["msg_id", "mdn", "final_pred", "source", "a2p_tags", "cluster_id"]
    ),
    lambda df: df.drop_duplicates(subset="message"),
)
