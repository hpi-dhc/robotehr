import dill
import numpy as np
from morpher.metrics import get_net_benefit_metrics
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(evaluation):
    precision, recall, _ = precision_recall_curve(
        evaluation["y_true"],
        evaluation["y_probs"]
    )
    auc_prc = auc(recall, precision)

    return {
        "average_precision": average_precision_score(
            evaluation["y_true"], evaluation["y_probs"]
        ),
        "f1": f1_score(evaluation["y_true"], evaluation["y_pred"]),
        "precision": precision_score(
            evaluation["y_true"], evaluation["y_pred"]
        ),
        "recall": recall_score(
            evaluation["y_true"], evaluation["y_pred"]
        ),
        "accuracy": accuracy_score(
            evaluation["y_true"], evaluation["y_pred"]
        ),
        "auc_roc": roc_auc_score(
            evaluation["y_true"], evaluation["y_probs"]
        ),
        "auc_prc": auc_prc,
        **evaluation
    }


def score_auc_nbc(training_result, tr_start=0.01, tr_end=0.99, tr_step=0.01, metric_type="treated"):
    auc_nbc = []
    outcome = dill.load(open(training_result.evaluation_path, 'rb'))
    for index, row in outcome.iterrows():
        tr_probs = np.arange(tr_start, tr_end + tr_step, tr_step)
        y_true = row.y_true
        y_probs = row.y_probs

        net_benefit, _ = get_net_benefit_metrics(
            y_true,
            y_probs,
            tr_probs,
            metric_type
        )
        auc_nbc.append(auc(tr_probs, net_benefit))
    scores = np.array(auc_nbc)
    return scores.mean(), scores.std()
