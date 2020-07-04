import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from robotehr.pipelines.training import load_features_and_transform
from robotehr.utils import FriendlyNamesConverter


def _quartile_information(row, quartile_limits):
    if row['calibrated_probs'] <= quartile_limits[0]:
        return "Q1"
    if (row['calibrated_probs'] > quartile_limits[0]) and (row['calibrated_probs'] <= quartile_limits[1]):
        return "Q2"
    if (row['calibrated_probs'] > quartile_limits[1]) and (row['calibrated_probs'] <= quartile_limits[2]):
        return "Q3"
    if (row['calibrated_probs'] > quartile_limits[2]) and (row['calibrated_probs'] <= quartile_limits[3]):
        return "Q4"
    if (row['calibrated_probs'] > quartile_limits[3]):
        return "Q5"


def _age_conversion(row):
    return row['age_in_days'] / 365


def make_risk_groups(predictor, data_loader, calibration_method="sigmoid"):
    X_raw, _ = load_features_and_transform(
        training_configuration=predictor.training_configuration,
        data_loader=data_loader,
        persist_data=False
    )

    X, y = predictor.get_features(), predictor.get_targets()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # fit and calibrate model on training data
    calibrator = CalibratedClassifierCV(
        base_estimator=predictor.clf.clf,
        cv="prefit",
        method=calibration_method
    )
    calibrator.fit(X_train, y_train)
    # evaluate the model
    y_hat = calibrator.predict_proba(X_test)[:,1]

    # find quartiles
    calibrated_probs = pd.Series(y_hat)
    quartile_limits = [calibrated_probs.quantile(x) for x in [i/100 for i in range(0, 100, int(100/5))][1:]]

    # build dataframe for downstream analysis
    df = X_raw.iloc[X_test.index]
    df = df[df.age_in_days < 50000]
    df['age'] = df.apply(_age_conversion, axis=1)
    df = df.reset_index().drop(columns=['index'])
    for c in df.columns:
        df[c] = df[c].astype(float)

    df['calibrated_probs'] = calibrated_probs
    df['risk group'] = df.apply(lambda x: _quartile_information(x, quartile_limits), axis=1)
    return df.sort_values(by="risk group")


def plot_risk_groups(df, features, friendly_names_converter=None, filename='', nrows=1, figsize=[12,3]):
    ncols = int(len(features) / nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.tight_layout(pad=3.0)

    for i in range(len(features)):
        row_index = int(i / ncols)
        col_index = i % int(len(features) / nrows)

        current_axis = ax[row_index][col_index] if nrows > 1 else ax[col_index]
        if df[features[i]].min() == 0 and df[features[i]].max() == 1:
            current_axis.set_ylim(bottom=-0.5, top=1.5)
        sns.violinplot(
            x="risk group",
            y=features[i],
            data=df,
            palette="muted",
            ax=current_axis
        )
        if friendly_names_converter:
            title = friendly_names_converter.get(features[i])
        else:
            title = features[i]
        if len(title) > 50:
            title = f'{title[:50]} ...'
        current_axis.set_title(f'{title}', fontsize=11)
        current_axis.set_xlabel('')
        current_axis.set_ylabel('')
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig
