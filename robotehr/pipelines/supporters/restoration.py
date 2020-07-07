import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from robotehr.api.training import get_training_configuration
from robotehr.pipelines.supporters.preprocessing import recursive_feature_elimination
from robotehr.pipelines.supporters.scoring import calculate_metrics


def restore_model(training_configuration, algorithm, sampler, run_rfe=True):
    df = pd.read_csv(training_configuration.training_data.path)
    # TODO: apply transformation here
    X, y = data_loader.transform(
        X=data.drop(columns=[target]),
        y=data[target]
    )

    X_train, X_test, y_train, y_test = train_test_split(X_supported, y, test_size=0.2, random_state=42)

    X_train, y_train = data_loader.transform_training_data(X_train, y_train)
    X_test, y_test = data_loader.transform_test_data(X_test, y_test)

    if rfe__run and algorithm().clf.__class__ != DummyClassifier:
        rfe = RFE(
            estimator=algorithm(),
            step=50,
        )

        X_train_supported = rfe.fit_transform(X_train, y_train)
        X_test_supported = rfe.transform(X_test)
    else:
        X_train_supported = X_train.copy()
        X_test_supported = X_test.copy()

    X_train_sampled, y_train_sampled = sampler().fit_resample(X_train_supported, y_train)
    clf = algorithm()
    clf.fit(X_train_sampled, y_train_sampled)

    auc_roc_score = clf.score_auroc(clf, X_test, y_test)

    print(f'Achieved an AUC of {auc_roc_score}')

    return {
        'clf': clf,
        'X_train': X_train_sampled,
        'y_train': y_train_sampled,
        'X_test': X_test_supported,
        'y_test': y_test
    }
