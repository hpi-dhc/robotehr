import json

import pandas as pd
from sqlalchemy import orm, asc, desc
from flask import request

from robotehr.models import engine, session
from robotehr.models.training import (
    TrainingConfiguration,
    TrainingPipeline,
    TrainingResult
)
from robotehr.api import app
from robotehr.api.helpers import assert_response_type, build_response, sort_and_filter

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

@app.route('/api/training/results')
def get_training_results(
    pipeline_id=None,
    columns=DEFAULT_COLUMNS,
    metrics=DEFAULT_METRICS,
    response_type="json",
    **kwargs
):
    assert_response_type(response_type)
    pipeline_id = pipeline_id or request.args.get('pipeline', type=int)
    assert pipeline_id is not None
    q = session.query(
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
    q = sort_and_filter(q, **kwargs)
    results = [row._asdict() for row in q]
    return build_response(results, response_type)


@app.route('/api/training/pipelines')
def get_training_pipelines(
    sort_by=TrainingPipeline.start_time,
    response_type="json",
    **kwargs
):
    assert_response_type(response_type)
    q = session.query(TrainingPipeline)
    q = sort_and_filter(q, sort_by=sort_by, **kwargs)
    return build_response(q.all(), response_type)
