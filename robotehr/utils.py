import dill
import json

import numpy as np
import requests
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


def http_post(url, data, headers={}):
    default_headers = {"Content-Type": "application/json"}
    response = requests.post(
        url, data=json.dumps(data), headers={**default_headers, **headers},
    )
    if response.status_code != 200:
        raise ValueError(
            "Request returned an error %s, the response is:\n%s"
            % (response.status_code, response.text)
        )
    return response


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


class DataLoader:
    def __init__(self, agg_func_regex, prepare_data_function):
        self.agg_func_regex = agg_func_regex
        self.prepare_data_function = prepare_data_function

    def transform(self, df, target):
        return self.prepare_data_function(df, target)

    def dump(self, path):
        return dill.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path):
        return dill.load(open(path, "rb"))


def score_auroc(estimator, X, y):
    """
    AUROC scoring method for estimators (classifiers) so we can use AUROC
    in our own definition.
    """
    y_pred = estimator.predict_proba(X)
    return roc_auc_score(y, y_pred[:, 1])


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
