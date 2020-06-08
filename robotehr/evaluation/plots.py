import dill

import matplotlib.pyplot as plt
from morpher.plots import plot_dc, plot_cc
from sklearn.model_selection import train_test_split


def clinical_usefulness_graph(training_results, metric_type="treated", filename="", label=""):
    fig = plt.figure(figsize=[12,8])
    for tr in training_results:
        label = label or tr.algorithm
        outcome = dill.load(open(tr.evaluation_path, 'rb'))
        for index, row in outcome.iterrows():

            plot_dc(
                results={
                    tr.algorithm: {
                        'y_true': row.y_true,
                        'y_pred': row.y_pred,
                        'y_probs': row.y_probs,
                        'label': f'{label} | (fold #{index})'
                    }
                },
                tr_start=0.01,
                tr_end=0.99,
                tr_step=0.01,
                metric_type=metric_type
            )
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig


def calibration_plot(predictor, name, **kwargs):
    train_data, test_data = train_test_split(predictor.get_data())

    plot_cc(
        models={name: predictor.clf},
        train_data=train_data,
        test_data=test_data,
        target=predictor.target,
        **kwargs
    )
