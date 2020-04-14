import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from robotehr.controllers import get_training_results


def average_metric_score_by_parameter(pipeline_id, metric, parameter):
    df = get_training_results(pipeline_id, sort_by=metric, columns=[parameter], metrics=[metric])
    df = df.groupby(parameter).mean().sort_values(by=metric, ascending=False).reset_index()
    return df


def average_metric_score(
    df,
    algorithm,
    metric,
    target,
    dimensions=['window_start_occurring', 'threshold_occurring']
):
    assert len(dimensions) == 2
    x_dimension, y_dimension = dimensions
    col_replace = {
        x_dimension: x_dimension.replace('_', ' ').title(),
        y_dimension: y_dimension.replace('_', ' ').title(),
        metric: metric.replace('_', ' ').upper()
    }
    data = df[
        (df.algorithm == algorithm)
        & (df.target == target)
    ][[
        *dimensions,
        metric
    ]].groupby(
        dimensions
    ).agg('mean').reset_index().rename(columns=col_replace)

    return data.pivot_table(
        index=col_replace[y_dimension],
        columns=col_replace[x_dimension],
        values=col_replace[metric]
    )


def performance_heatmap(
    pipeline_id,
    algorithms,
    metrics,
    target,
    dimensions=['window_start_occurring', 'threshold_occurring'],
    annot=False,
    filename=None
):
    df = get_training_results(pipeline_id, metrics=metrics)
    fig = plt.figure(figsize=(10 * len(algorithms), 10 * len(metrics)))

    vmins = {}
    vmaxs = {}
    for j in range(len(metrics)):
        vmin = 1
        vmax = 0
        for i in range(len(algorithms)):
            data = average_metric_score(df, algorithms[i], metrics[j], target, dimensions)
            vmin = min(vmin, math.floor(data.min().min() * 10) / 10.0)
            vmax = max(vmax, math.ceil(data.max().max() * 10) / 10.0)
        vmins[metrics[j]] = vmin
        vmaxs[metrics[j]] = vmax

    n = 0
    for j in range(len(metrics)):
        for i in range(len(algorithms)):
            data = average_metric_score(df, algorithms[i], metrics[j], target, dimensions)
            plt.subplot(len(metrics), len(algorithms), n + 1)
            cbar = i + 1 == len(algorithms)
            ax = sns.heatmap(
                data, vmin=vmins[metrics[j]], cmap="YlGnBu",
                vmax=vmaxs[metrics[j]], cbar=cbar, annot=annot,
                linewidths=2, square=False,
            )
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            plt.yticks(rotation=0, fontsize=14)
            plt.xticks(rotation=0, fontsize=14)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)
            metric = metrics[j].replace('_', ' ').upper()
            plt.title(
                f'Mean {metric} Score for {algorithms[i]}',
                fontsize=20
            )
            n += 1
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig


def get_quandrant_averages(
    pipeline_id,
    algorithm,
    metric,
    target,
    lead_condition_type='occurring'
):
    assert lead_condition_type in ['occurring', 'numeric']

    threshold_column = f'threshold_{lead_condition_type}'
    window_start_column = f'window_start_{lead_condition_type}'
    df = get_training_results(pipeline_id, metrics=metrics)
    data = df[
        (df.algorithm == algorithm)
        & (df.target == target)
    ][[
        threshold_column, window_start_column, metric
    ]]
    limits = {
        threshold_column: data[threshold_column].median(),
        window_start_column: data[window_start_column].median()
    }
    quadrants = {
        'Q1': data[
            (data[threshold_column] <= limits[threshold_column])
            & (data[window_start_column] >= limits[window_start_column])
        ][metric],
        'Q2': data[
            (data[threshold_column] < limits[threshold_column])
            & (data[window_start_column] < limits[window_start_column])
        ][metric],
        'Q3': data[
            (data[threshold_column] > limits[threshold_column])
            & (data[window_start_column] < limits[window_start_column])
        ][metric],
        'Q4': data[
            (data[threshold_column] >= limits[threshold_column])
            & (data[window_start_column] >= limits[window_start_column])
        ][metric],
    }
    results = {
        'quadrants': dict([
            (q, {'mean': v.mean(), 'std': v.std()})
            for (q, v) in quadrants.items()
        ]),
        'limits': limits
    }
    return results


def quadrant_averages_plot(
    pipeline_id,
    algorithms,
    target,
    metric,
    filename=None
):
    df = pd.DataFrame(columns=['algorithm', 'quadrant', 'mean', 'std'])

    for algorithm in algorithms:
        d = get_quandrant_averages(
            pipeline_id,
            algorithm,
            metric,
            target
        )['quadrants']
        for quadrant, value in d.items():
            df = df.append({
                'algorithm': algorithm,
                'quadrant': quadrant,
                'mean': value['mean'],
                'std': value['std']
            }, ignore_index=True)

    cat = "quadrant"
    subcat = "algorithm"
    val = "mean"
    err = "std"
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (
        np.arange(len(subx)) - np.arange(len(subx)).mean()
    ) / (len(subx) + 1.)
    width = np.diff(offsets).mean()

    fig = plt.figure()
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width,
                label=gr, yerr=dfg[err].values)
    plt.xlabel(cat.capitalize())
    plt.ylabel(metric)
    plt.xticks(x, u)
    plt.legend(loc=4)
    plt.title(f'Average {metric} Score per Quadrant')

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig


def performance_diff_heatmap(
    pipeline_ids,
    algorithms,
    metrics,
    target,
    annot=False,
    filename=None
):
    df = [
        get_training_results(pipeline_ids[0], metrics=metrics),
        get_training_results(pipeline_ids[1], metrics=metrics)
    ]
    fig = plt.figure(figsize=(10 * len(algorithms), 10 * len(metrics)))

    vmins = {}
    vmaxs = {}
    for j in range(len(metrics)):
        vmin = 0
        vmax = 0
        for i in range(len(algorithms)):
            data = average_metric_score(df[1], algorithms[i], metrics[j], target) - average_metric_score(df[0], algorithms[i], metrics[j], target)
            vmax = max(vmax, data.abs().max().max())
        vmaxs[metrics[j]] = vmax
    n = 0
    for j in range(len(metrics)):
        for i in range(len(algorithms)):
            data = (
                average_metric_score(
                    df[1],
                    algorithms[i],
                    metrics[j],
                    target
                ) - average_metric_score(
                    df[0],
                    algorithms[i],
                    metrics[j],
                    target
                )
            )
            data = data.dropna(axis=0, how='all').dropna(axis=1, how='all')
            plt.subplot(len(metrics), len(algorithms), n + 1)
            cbar = i + 1 == len(algorithms)
            ax = sns.heatmap(
                data,
                vmin=-vmaxs[metrics[j]],
                vmax=vmaxs[metrics[j]],
                cmap=sns.diverging_palette(10, 240, as_cmap=True),
                cbar=cbar, annot=annot,
                linewidths=2, square=False,
            )
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            plt.yticks(rotation=0, fontsize=14)
            plt.xticks(rotation=0, fontsize=14)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)
            plt.title(
                f'Mean {metrics[j]} change for {algorithms[i]}',
                fontsize=20
            )
            n += 1
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig
