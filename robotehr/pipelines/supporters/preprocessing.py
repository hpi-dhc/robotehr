from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from robotehr.pipelines.supporters.plots import plot_rfe


def recursive_feature_elimination(X, y, algorithm, step_size=50, create_figure=False, filename="", n_splits=5):
    estimator = algorithm()
    rfecv = RFECV(
        estimator=estimator,
        step=step_size,
        cv=StratifiedKFold(n_splits=n_splits),
        scoring=estimator.score_auroc,
        verbose=1,
        n_jobs=-1
    )
    rfecv.fit(X, y)

    if create_figure:
        fig = plot_rfe(rfecv, X, step_size, filename)
    else:
        fig = {}

    return {
        'n_features': rfecv.n_features_,
        'X_supported': X[X.columns[rfecv.support_]],
        'y': y,
        'figure': fig,
        'rfecv': rfecv,
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
