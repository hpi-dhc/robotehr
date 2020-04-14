import json

import pandas as pd
from sqlalchemy import orm, asc, desc

from robotehr.models import engine
from robotehr.models.training import (
    TrainingConfiguration,
    TrainingPipeline,
    TrainingResult
)

DEFAULT_COLUMNS = [
    TrainingConfiguration.target,
    TrainingConfiguration.threshold_numeric,
    TrainingConfiguration.window_start_numeric,
    TrainingConfiguration.threshold_occurring,
    TrainingConfiguration.window_start_occurring,
    TrainingResult.algorithm,
    TrainingResult.sampler
]

DEFAULT_METRICS = [
    TrainingResult.average_precision_mean,
    TrainingResult.f1_mean,
    TrainingResult.precision_mean,
    TrainingResult.recall_mean,
    TrainingResult.accuracy_mean,
    TrainingResult.auc_roc_mean
]


def get_training_results(
    pipeline_id,
    sort_by=None,
    n_rows=None,
    columns=DEFAULT_COLUMNS,
    metrics=DEFAULT_METRICS
):
    q = orm.Query(
        TrainingResult
    ).join(
        TrainingConfiguration,
        TrainingResult.training_configuration_id == TrainingConfiguration.id
    ).filter(
        TrainingConfiguration.training_pipeline_id == pipeline_id
    ).with_entities(
        *columns,
        *metrics
    )
    if sort_by:
        q = q.order_by(desc(sort_by))
    if n_rows:
        q = q.limit(n_rows)
    return pd.read_sql(q.statement, engine)


def list_training_pipelines(
    sort_by=TrainingPipeline.start_time,
    sort_order="desc",
    n_rows=None
):
    q = orm.Query(
        TrainingPipeline
    ).with_entities(
        TrainingPipeline.id,
        TrainingPipeline.comment,
        TrainingPipeline.version,
        TrainingPipeline.start_time,
        TrainingPipeline.end_time,
    )
    if sort_by:
        if sort_order == "asc":
            q = q.order_by(asc(sort_by))
        else:
            q = q.order_by(desc(sort_by))
    if n_rows:
        q = q.limit(n_rows)
    results = pd.read_sql(q.statement, engine)
    results['run_time'] = results['end_time'] - results['start_time']
    return results
