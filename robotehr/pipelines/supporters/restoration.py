import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from robotehr.api.training import get_training_configuration
from robotehr.pipelines.supporters.preprocessing import recursive_feature_elimination
from robotehr.pipelines.supporters.scoring import calculate_metrics


def restore_model(training_configuration, algorithm, sampler):
    df = pd.read_csv(training_configuration.training_data.path)
    # TODO: apply transformation here

    X = df.drop(columns=[training_configuration.target])
    y = df[training_configuration.target]

    result = recursive_feature_elimination(
        X=X,
        y=y,
        step_size=50,
        n_splits=5,
        algorithm=algorithm,
        filename="",
        create_figure=False
    )
    X_supported = result['X_supported']

    X_train, X_test, y_train, y_test = train_test_split(X_supported, y, test_size=0.2, random_state=42)
    X_train_sampled, y_train_sampled = sampler().fit_resample(X_train, y_train)
    clf = algorithm()
    clf.fit(X_train_sampled, y_train_sampled)

    auc_roc_score = clf.score_auroc(clf, X_test, y_test)

    print(f'Achieved an AUC of {auc_roc_score}')

    return {
        'clf': clf,
        'X': X_supported,
        'y': y
    }
