from itertools import product

import pandas as pd
from fiber.config import OCCURRENCE_INDEX
from fiber.dataframe.helpers import get_name_for_interval
from fiber.dataframe import merge_to_base, column_threshold_clip
from fiberutils.cohort_utils import threshold_clip_time_series, pivot_time_series
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold

from robotehr.config import WEBHOOK_URL
from robotehr.models import session
from robotehr.models.data import Feature
from robotehr.models.training import TrainingConfiguration, TrainingData, TrainingPipeline, TrainingResult
from robotehr.pipelines.supporters.preprocessing import recursive_feature_elimination
from robotehr.pipelines.supporters.scoring import calculate_metrics
from robotehr.utils import http_post


def load_features_and_transform(
    training_configuration,
    data_loader,
    bin_size=30,
    persist_data=True
):
    target = training_configuration.target
    cohort = training_configuration.training_pipeline.cohort.get_fiber()
    onset_df = training_configuration.training_pipeline.onset_dataframe.get_df(target)
    feature_pipeline = training_configuration.training_pipeline.feature_pipeline

    numeric_feature_dfs = []
    if training_configuration.feature_type_numeric == 'numeric_binned':
        window = training_configuration.window_start_numeric, training_configuration.window_end_numeric
        for w in range(window[0], window[1], bin_size):
            numeric_feature_objs = session.query(Feature).filter_by(
                feature_pipeline=feature_pipeline,
                feature_type=training_configuration.feature_type_numeric,
                window_start=w,
                window_end=w + bin_size
            ).all()
            cur_numeric_dfs = [pd.read_csv(f.path) for f in numeric_feature_objs]
            for df in cur_numeric_dfs:
                df.set_index(OCCURRENCE_INDEX, inplace=True)
                new_cols = [get_name_for_interval(c, [w, w + bin_size]) for c in df.columns]
                df.columns = new_cols
                df.reset_index(inplace=True)
            numeric_feature_dfs += cur_numeric_dfs
        _, occurring_feature_dfs = feature_pipeline.get_features(training_configuration)

    else:
        numeric_feature_dfs, occurring_feature_dfs = feature_pipeline.get_features(training_configuration)

    cohort.occurrences.medical_record_number = cohort.occurrences.medical_record_number.astype(int)
    onset_df.medical_record_number = onset_df.medical_record_number.astype(int)

    if training_configuration.feature_type_numeric == "numeric_time_series":
        pivoted_dfs = []
        for df in numeric_feature_dfs:
            numeric_df = threshold_clip_time_series(
                df=df,
                cohort=cohort,
                threshold=training_configuration.threshold_numeric
            )
            pivoted_dfs.append(pivot_time_series(
                cohort=cohort,
                onset_df=onset_df,
                df=numeric_df,
            ))
        numeric_df = merge_to_base(
            cohort.occurrences,
            pivoted_dfs
        )

    else:
        numeric_df = merge_to_base(
            cohort.occurrences,
            [
                x.filter(
                    regex=(data_loader.column_selector + "|medical_record_number|age_in_days")
                )
                for x in numeric_feature_dfs
            ]
        )
        numeric_df = column_threshold_clip(
            df=numeric_df,
            threshold=training_configuration.threshold_numeric
        )

    occurring_df = merge_to_base(
        cohort.occurrences,
        [
            x.filter(
                regex=(data_loader.column_selector + "|medical_record_number|age_in_days")
            )
            for x in occurring_feature_dfs
        ]
    )
    occurring_df = column_threshold_clip(
        df=occurring_df,
        threshold=training_configuration.threshold_occurring
    )

    cohort.occurrences.medical_record_number = cohort.occurrences.medical_record_number.astype(str)
    numeric_df.medical_record_number = numeric_df.medical_record_number.astype(str)
    occurring_df.medical_record_number = occurring_df.medical_record_number.astype(str)
    onset_df.medical_record_number = onset_df.medical_record_number.astype(str)

    ## Merge to cohort data and use user's data loader
    data = cohort.merge_patient_data(
        onset_df,
        numeric_df,
        occurring_df,
    )

    ## persist training data
    if persist_data:
        TrainingData.persist(
            training_configuration=training_configuration,
            data=data
        )
    X, y = data_loader.transform(
        X=data.drop(columns=[target]),
        y=data[target]
    )
    return X, y


def train_iteration(
    threshold_numeric,
    observation_window_numeric,
    threshold_occurring,
    observation_window_occurring,
    target,
    algorithms,
    samplers,
    training_pipeline,
    data_loader,
    feature_type_occurring,
    feature_type_numeric,
    bin_size,
    rfe__run,
    rfe__step_size,
    persist_data
):
    training_configuration = TrainingConfiguration.persist(
        training_pipeline=training_pipeline,
        threshold_occurring=threshold_occurring,
        window_start_occurring=observation_window_occurring[0],
        window_end_occurring=observation_window_occurring[1],
        feature_type_occurring=feature_type_occurring,
        threshold_numeric=threshold_numeric,
        window_start_numeric=observation_window_numeric[0],
        window_end_numeric=observation_window_numeric[1],
        feature_type_numeric=feature_type_numeric,
        target=target
    )

    # Load data and transform
    X, y = load_features_and_transform(
        training_configuration=training_configuration,
        data_loader=data_loader,
        bin_size=bin_size,
        persist_data=persist_data
    )

    for algorithm in algorithms:
        for sampler in samplers:
            metrics = []
            data_statistics = []
            cv = StratifiedKFold(n_splits=5)
            for train_idx, test_idx, in cv.split(X, y):
                X_train, y_train = X.loc[train_idx], y.loc[train_idx]
                X_test, y_test = X.loc[test_idx], y.loc[test_idx]

                X_train, y_train = data_loader.transform_training_data(X_train, y_train)
                X_test, y_test = data_loader.transform_test_data(X_test, y_test)

                if rfe__run and algorithm().clf.__class__ != DummyClassifier:
                    rfe = RFE(
                        estimator=algorithm(),
                        step=rfe__step_size,
                    )

                    X_train_supported = rfe.fit_transform(X_train, y_train)
                    X_test_supported = rfe.transform(X_test)
                else:
                    X_train_supported = X_train.copy()
                    X_test_supported = X_test.copy()

                X_train_sampled, y_train_sampled = sampler().fit_resample(X_train_supported, y_train)
                clf = algorithm()
                clf.fit(X_train_sampled, y_train_sampled)

                evaluation = {
                    'y_true': y_test,
                    'y_pred': clf.predict(X_test_supported),
                    'y_probs': clf.predict_proba(X_test_supported)[:, 1],
                }
                metrics.append(calculate_metrics(evaluation))
                data_statistics.append({
                    'num_test_rows': len(X_test),
                    'num_train_rows': len(X_train),
                    'num_features': len(X_train.columns),
                    'incidence_rate_train': y_train.sum() / len(X_train),
                    'incidence_rate_test': y_test.sum() / len(X_test),
                    'num_sampled_train_rows': len(X_train_sampled),
                    'incidence_rate_sampled_train': y_train_sampled.sum() / len(X_train_sampled),
                })
            TrainingResult.persist(
                training_configuration=training_configuration,
                metrics=pd.DataFrame(metrics),
                data_statistics=pd.DataFrame(data_statistics),
                algorithm=algorithm,
                sampler=sampler
            )


def execute(
    comment,
    version,
    cohort,
    onset_dataframe,
    feature_pipeline,
    data_loader,
    observation_iterator,
    targets,
    algorithms,
    samplers,
    feature_type_occurring,
    feature_type_numeric,
    bin_size=30,
    rfe__run=False,
    rfe__step_size=50,
    persist_data=True
):
    training_pipeline = TrainingPipeline.create(
        comment=comment,
        version=version,
        cohort=cohort,
        onset_dataframe=onset_dataframe,
        data_loader=data_loader,
        feature_pipeline=feature_pipeline
    )

    i = 0
    for (
        threshold_numeric,
        observation_window_numeric,
        threshold_occurring,
        observation_window_occurring
    ) in observation_iterator:
        for target in targets:
            train_iteration(
                threshold_numeric=threshold_numeric,
                observation_window_numeric=observation_window_numeric,
                threshold_occurring=threshold_occurring,
                observation_window_occurring=observation_window_occurring,
                target=target,
                algorithms=algorithms,
                samplers=samplers,
                training_pipeline=training_pipeline,
                data_loader=data_loader,
                feature_type_occurring=feature_type_occurring,
                feature_type_numeric=feature_type_numeric,
                bin_size=bin_size,
                rfe__run=rfe__run,
                rfe__step_size=rfe__step_size,
                persist_data=persist_data
            )
            i += 1
            if i % 10 == 0:
                http_post(
                    WEBHOOK_URL,
                    {"text": f"{i} iterations of training for '{comment}' in version {version} done."} # noqa
                )

    training_pipeline.end_pipeline()
    http_post(
        WEBHOOK_URL,
        {"text": f"Training done in {i} iterations for '{comment}' in version {version}."} # noqa
    )

    return training_pipeline
