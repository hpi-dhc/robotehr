from itertools import product

from fiberutils.cohort_utils import get_time_series

from robotehr.config import WEBHOOK_URL
from robotehr.models.cohort import Cohort
from robotehr.models.data import Feature, FeaturePipeline
from robotehr.utils import http_post


def _build_default_configs(
    condition: list,
    min_threshold: float,
    windows: list,
    agg_func: dict,
    **kwargs,
):
    configs = []
    for window in windows:
        configs.append({
            "condition": condition,
            "threshold": min_threshold,
            "window": window,
            "pivot_table_kwargs": {
                "columns": ["description"],
                "aggfunc": agg_func,
            }
        })
    return configs


def _build_binned_configs(
    condition: list,
    min_threshold: float,
    windows: list,
    agg_func: dict,
    bin_size=30,
    **kwargs
):
    configs = []
    for window in windows:
        for w in range(window[0], window[1], bin_size):
            new_conf = {
                "condition": condition,
                "threshold": min_threshold,
                "window": [w, w + bin_size],
                "pivot_table_kwargs": {
                    "columns": ["description"],
                    "aggfunc": agg_func,
                }
            }
            if not new_conf in configs:
                configs.append(new_conf)
    return configs


def _build_time_series_configs(
    condition: list,
    windows: list,
    min_threshold: int,
    **kwargs
):
    configs = []
    for window in windows:
        configs.append({
            "condition": condition,
            "window": window,
            "threshold": min_threshold
        })
    return configs


def pivot_data(
    pivot_configuration,
    min_threshold,
    cohort,
    feature_pipeline
):
    feature_type = pivot_configuration["feature_type"]
    fiber_cohort = Cohort.load_fiber(cohort.id)

    if feature_type in ["numeric", "occurring"]:
        configs = _build_default_configs(
            min_threshold=min_threshold,
            **pivot_configuration
        )
    elif feature_type in ["numeric_binned", "occurring_binned"]:
        configs = _build_binned_configs(
            min_threshold=min_threshold,
            **pivot_configuration
        )
    elif feature_type in ["numeric_time_series"]:
        configs = _build_time_series_configs(
            min_threshold=min_threshold,
            **pivot_configuration
        )

    # TODO: parallelize this step
    for c in configs:
        if feature_type in ["numeric", "occurring", "numeric_binned", "occurring_binned"]:
            df = fiber_cohort.pivot_all_for(**c).reset_index()
        elif feature_type in ["numeric_time_series"]:
            df = get_time_series(
                cohort=fiber_cohort,
                condition=c["condition"],
                window=c["window"],
                threshold=min_threshold
            ).reset_index()

        Feature.persist(
            df=df,
            min_threshold=min_threshold,
            feature_pipeline=feature_pipeline,
            feature_type=feature_type,
            **c
        )


def execute(configuration, comment, version, cohort):
    feature_pipeline = FeaturePipeline.persist(comment=comment, version=version, cohort=cohort)

    min_threshold = configuration["min_threshold"]
    pivot_configurations = configuration["pivot_configurations"]
    number_of_configs = len(pivot_configurations)
    for i in range(number_of_configs):
        pivot_data(
            pivot_configuration=pivot_configurations[i],
            min_threshold=min_threshold,
            cohort=cohort,
            feature_pipeline=feature_pipeline
        )

    # End pipeline
    feature_pipeline.end_pipeline()
    http_post(
        WEBHOOK_URL,
        {"text": f"Features extracted & persisted for pipeline '{comment}' in version {version}."} # noqa
    )
    return feature_pipeline
