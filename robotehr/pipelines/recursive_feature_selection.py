import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

from robotehr.models.training import TrainingConfiguration
from robotehr.utils import calculate_metrics, DataLoader, score_auroc

def execute(X, y, algorithm, step_size, create_figure=False, filename="", n_splits=5):
    estimator = algorithm()
    rfecv = RFECV(
        estimator=estimator,
        step=step_size,
        cv=StratifiedKFold(n_splits=n_splits),
        scoring=estimator.score_auroc,
        verbose=2,
        n_jobs=-1
    )
    rfecv.fit(X, y)

    steps = list(range(len(X.columns), 0, -step_size))
    if 1 not in steps:
        steps.append(1)
    steps.reverse()
    if create_figure:
        fig = plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validated auroc score")
        plt.plot(steps, rfecv.grid_scores_)
        plt.title("")
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
    else:
        fig = {}

    return {
        'n_features': rfecv.n_features_,
        'X_supported': X[X.columns[rfecv.support_]],
        'y': y,
        'figure': fig,
        'rfecv': rfecv,
    }
