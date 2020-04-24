def clinical_usefulness_graph(training_results, metric_type="treated", filename=""):
    fig = plt.figure(figsize=[16,16])
    for tr in training_results:
        outcome = dill.load(open(tr.evaluation_path, 'rb'))
        for index, row in outcome.iterrows():
            plot_dc(
                results={
                    tr.sampler: {
                        'y_true': row.y_true,
                        'y_pred': row.y_pred,
                        'y_probs': row.y_probs,
                        'label': f'{tr} | (fold #{index})'
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
