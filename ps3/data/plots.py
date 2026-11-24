import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_weighted_avg_claims(
    df,
    group_col="BonusMalus",
    claims_col="PurePremium",
    weight_col="Exposure",
    pred_cols=None,
    bins=None,
    ax=None,
):
    """Plot the exposure-weighted average claims per group.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : Column to group by
    claims_col : Column with the observed claims per unit of exposure, by default - "PurePremium".
    weight_col : Column with the weights, by default "Exposure".
    pred_cols : dict, optional
        Model predictions to plot alongside the observed claims, as
        ``{label: column name}``. They are averaged in the same way.
    bins : int or sequence of float, optional
        If given, ``group_col`` is binned with ``pd.cut`` and the groups are the
        bins (plotted at their midpoints). By default every distinct value of
        ``group_col`` is its own group.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if not given.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    pred_cols = {} if pred_cols is None else dict(pred_cols)
    value_cols = {"Observed": claims_col, **pred_cols}

    groups = df[group_col] if bins is None else pd.cut(df[group_col], bins=bins)

    weights = df[weight_col]
    weighted = pd.DataFrame(
        {label: df[col] * weights for label, col in value_cols.items()}
    )
    weighted[weight_col] = weights

    sums = weighted.groupby(groups, observed=True).sum()
    averages = sums[list(value_cols)].div(sums[weight_col], axis=0)

    x = averages.index
    if bins is not None:
        x = x.map(lambda interval: interval.mid)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, averages["Observed"], marker="o", linestyle="-", label="Observed")
    for label in pred_cols:
        ax.plot(x, averages[label], linestyle="--", label=label)

    ax.set(
        title=f"Exposure-weighted average claims by {group_col}",
        xlabel=group_col,
        ylabel=f"Average {claims_col} (weighted by {weight_col})",
    )
    ax.legend()
    return ax
