import dill
import numpy as np
import pandas as pd
import scipy.stats as st

from robotehr.api.training import get_training_configuration
from robotehr.models.training import TrainingResult
from robotehr.pipelines.supporters.scoring import (
    calculate_metrics,
    advertised_metrics
)


def _confidence_interval(results, interval=.95):
    mean = np.mean(results)
    interval = st.t.interval(
        interval,
        len(results) - 1,
        loc=mean,
        scale=st.sem(results)
    )
    return mean, mean - interval[0]


def calculate_confidence_interval(
    pipeline_id,
    config,
    algorithm,
    sampler,
    metric
):
    assert metric in advertised_metrics()
    tc = get_training_configuration(
        pipeline_id=pipeline_id,
        config=config,
        response_type="object"
    )
    tr = TrainingResult.load_by_config(
        training_configuration=tc,
        algorithm=algorithm,
        sampler=sampler
    )
    results = dill.load(open(tr.evaluation_path, 'rb'))

    r = []
    for _, row in results.iterrows():
        r.append(calculate_metrics(row)[metric])

    return _confidence_interval(r)
