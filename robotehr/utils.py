import dill
import json

import requests
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
