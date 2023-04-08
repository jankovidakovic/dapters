from functools import partial

import numpy as np

def precision(*, true_positives, false_positives):
    if np.isclose(true_positives + false_positives, 0.0):
        return 0
    return true_positives / (true_positives + false_positives)

def recall(*, true_positives, false_negatives):
    if np.isclose(true_positives + false_negatives, 0.0):
        return 0
    return true_positives / true_positives + false_negatives


def f_score(*, true_positives, false_positives, false_negatives, beta=1):
    p = precision(true_positives=true_positives, false_positives=false_positives)
    r = recall(true_positives=true_positives, false_negatives=false_negatives)
    denominator = beta ** 2 * p + r
    if np.isclose(denominator, 0.0):
        return 0

    return p * r * (1 + beta ** 2) / denominator

f1_score = partial(
    f_score,
    beta=1
)

f2_score = partial(
    f_score,
    beta=2
)

f0_5_score = partial(
    f_score,
    beta=0.5
)

def macro_f_score(conf_matrix: np.array, f_score_function = f1_score):
    per_label_f1_scores = np.zeros(conf_matrix.shape[0])
    for i, per_label_confmat in enumerate(conf_matrix):
        true_negatives = per_label_confmat[0, 0]
        false_negatives = per_label_confmat[1, 0]
        true_positives = per_label_confmat[1, 1]
        false_positives = per_label_confmat[0, 1]

        per_label_f1_scores[i] = f_score_function(true_positives=true_positives, false_positives=false_positives, false_negatives=false_negatives)

    return np.mean(per_label_f1_scores)


def weighted_f_score(conf_matrix, f_score_function = f1_score):
    per_label_f_scores = np.zeros((conf_matrix.shape[0]))
    class_counts = np.zeros((conf_matrix.shape[0]))
    for i, per_label_confmat in enumerate(conf_matrix):
        true_negatives = per_label_confmat[0, 0]
        false_negatives = per_label_confmat[1, 0]
        true_positives = per_label_confmat[1, 1]
        false_positives = per_label_confmat[0, 1]

        per_label_f_scores[i] = f_score_function(true_positives=true_positives, false_positives=false_positives, false_negatives=false_negatives)
        class_counts[i] = true_positives + false_positives + true_negatives + false_negatives

    return np.average(per_label_f_scores, weights=class_counts / np.sum(class_counts))


def micro_f_score(conf_matrix: np.array, f_score_function = f1_score):
    true_positives = np.sum(conf_matrix[:, 1, 1])
    false_positives = np.sum(conf_matrix[:, 0, 1])
    false_negatives = np.sum(conf_matrix[:, 1, 0])

    return f_score_function(true_positives=true_positives, false_positives=false_positives, false_negatives=false_negatives)


def multilabel_classification_report(conf_matrix, label_names, per_class_metrics: bool = False):
    def per_label_metrics(label_index):
        true_positives = conf_matrix[label_index, 1, 1]
        false_positives = conf_matrix[label_index, 0, 1]
        true_negatives = conf_matrix[label_index, 0, 0]
        false_negatives = conf_matrix[label_index, 1, 0]
        num_examples = true_positives + false_positives + true_negatives + false_negatives

        return {
            "true_positives": int(true_positives),
            "true_negatives": int(true_negatives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "num_examples": int(num_examples),
            "precision": precision(true_positives=true_positives, false_positives=false_positives),
            "recall": recall(true_positives=true_positives, false_negatives=false_negatives),
            "f1_score": f1_score(true_positives=true_positives, false_positives=false_positives, false_negatives=false_negatives),
            "f2_score": f2_score(true_positives=true_positives, false_positives=false_positives, false_negatives=false_negatives),
            "f0_5_score": f0_5_score(true_positives=true_positives, false_positives=false_positives, false_negatives=false_negatives)
        }

    metrics = {
        "true_positives": np.sum(conf_matrix[:, 1, 1]),
        "true_negatives": np.sum(conf_matrix[:, 0, 1]),
        "false_positives": np.sum(conf_matrix[:, 0, 0]),
        "false_negatives": np.sum(conf_matrix[:, 1, 0]),
        "macro_f1": macro_f_score(conf_matrix),
        "macro_f2": macro_f_score(conf_matrix, f_score_function=f2_score),
        "macro_f0_5": macro_f_score(conf_matrix, f_score_function=f0_5_score),
        "micro_f1": micro_f_score(conf_matrix),
        "micro_f2": micro_f_score(conf_matrix, f_score_function=f2_score),
        "micro_f0_5": micro_f_score(conf_matrix, f_score_function=f0_5_score),
        "weighted_f1": weighted_f_score(conf_matrix),
        "weighted_f2": weighted_f_score(conf_matrix, f_score_function=f2_score),
        "weighted_f0_5": weighted_f_score(conf_matrix, f_score_function=f0_5_score),
    }

    if per_class_metrics:
        metrics["per_class_metrics"] = {
            label: per_label_metrics(i) for i, label in enumerate(label_names)
        }

    return metrics