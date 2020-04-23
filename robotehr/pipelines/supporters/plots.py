import matplotlib.pyplot as plt

def plot_rfe(rfecv, X, step_size, filename):
    steps = list(range(len(X.columns), 0, -step_size))
    if 1 not in steps:
        steps.append(1)
    steps.reverse()
    fig = plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validated auroc score")
    plt.plot(steps, rfecv.grid_scores_)
    plt.title("Recursive Feature Elimination")
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig
