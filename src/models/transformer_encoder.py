import logging

from transformers import AutoModelForSequenceClassification


logger = logging.getLogger(__name__)


def get_transformer_model(args):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=args.num_labels,
        cache_dir=args.cache_dir,
        problem_type="multi_label_classification"
    )

    return model
