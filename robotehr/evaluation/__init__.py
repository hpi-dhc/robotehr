import dill
import numpy as np
import pandas as pd
import scipy.stats as st

from robotehr.api.training import (
    get_training_configuration,
    get_training_results,
)
from robotehr.models.training import TrainingResult
from robotehr.pipelines.supporters.scoring import (
    calculate_metrics,
    advertised_metrics,
)


def _confidence_interval(results, interval=0.95):
    mean = np.mean(results)
    interval = st.t.interval(
        interval, len(results) - 1, loc=mean, scale=st.sem(results)
    )
    return mean, mean - interval[0]


def calculate_confidence_interval(
    pipeline_id, config, algorithm, sampler, metric
):
    assert metric in advertised_metrics()
    tc = get_training_configuration(
        pipeline_id=pipeline_id, config=config, response_type="object"
    )
    tr = TrainingResult.load_by_config(
        training_configuration=tc, algorithm=algorithm, sampler=sampler
    )
    results = dill.load(open(tr.evaluation_path, "rb"))

    r = []
    for _, row in results.iterrows():
        r.append(calculate_metrics(row)[metric])

    return _confidence_interval(r)


def print_best_model_with_confidence(
    pipeline_id, metrics=["auc_roc", "specificity", "sensitivity", "ppv", "npv"]
):
    best_model = get_training_results(
        sort_by="auc_roc_mean", pipeline_id=pipeline_id, response_type="pandas"
    )[
        ["algorithm", "sampler", "threshold_numeric", "window_start_numeric"]
    ].iloc[
        0
    ]

    config = {
        "threshold_numeric": float(best_model["threshold_numeric"]),
        "window_start_numeric": int(best_model["window_start_numeric"]),
    }
    algorithm = best_model["algorithm"]
    sampler = best_model["sampler"]
    ow = int(-1 * config["window_start_numeric"] / 30)
    print(
        f'{algorithm} & {sampler} & {ow} & {config["threshold_numeric"]:.2f} & ',
        end="",
    )
    for m in metrics:
        interval = calculate_confidence_interval(
            pipeline_id=pipeline_id,
            config=config,
            algorithm=algorithm,
            sampler=sampler,
            metric=m,
        )
        print(
            f"$ {interval[0]:.2f}\pm{interval[1]:.2f} $ & ".replace("0.", "."),
            end="",
        )
