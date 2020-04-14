import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer

from robotehr.models.training import TrainingConfiguration
from robotehr.pipelines import recursive_feature_selection
from robotehr.pipelines.training import load_features_and_transform
from robotehr.utils import calculate_metrics, DataLoader, score_auroc


def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


def execute(training_pipeline_id, config, sampler, algorithm, n_splits=5, param_grid={}, rfe__run=False, rfe__step_size=20):
    training_configuration = TrainingConfiguration.load_by_config(training_pipeline_id=training_pipeline_id, config=config)
    data_loader = DataLoader.load(training_configuration.training_pipeline.data_loader_path)
    X, y = load_features_and_transform(
        training_configuration=training_configuration,
        data_loader=data_loader
    )

    if rfe__run and algorithm().clf.__class__ != DummyClassifier:
        result = recursive_feature_selection.execute(
            X=X,
            y=y,
            step_size=rfe__step_size,
            n_splits=5,
            algorithm=algorithm,
        )
        X_supported = result['X_supported']
    else:
        X_supported = X.copy()

    metrics = []
    cv = StratifiedKFold(n_splits=n_splits)
    for train_idx, test_idx, in cv.split(X_supported, y):
        X_train, y_train = X_supported.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X_supported.loc[test_idx], y.loc[test_idx]

        X_train_sampled, y_train_sampled = sampler().fit_resample(X=X_train, y=y_train)

        estimator = algorithm()
        cv = GridSearchCV(
            estimator=estimator,
            cv=5,
            n_jobs=-1,
            scoring=estimator.score_auroc,
            param_grid=param_grid,
            verbose=1
        )
        cv.fit(X_train, y_train)

        evaluation = {
            'y_true': y_test,
            'y_pred': cv.predict(X_test),
            'y_probs': cv.predict_proba(X_test)[:, 1],
        }
        metrics.append([calculate_metrics(evaluation), cv])
    return metrics, X_supported
