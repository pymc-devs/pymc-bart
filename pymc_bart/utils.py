"""Utility function for variable selection and bart interpretability."""

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from pytensor.tensor.var import Variable
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.stats import norm, pearsonr


def _sample_posterior(all_trees, X, rng, size=None, excluded=None):
    """
    Generate samples from the BART-posterior.

    Parameters
    ----------
    all_trees : list
        List of all trees sampled from a posterior
    X : array-like
        A covariate matrix. Use the same used to fit BART for in-sample predictions or a new one for
        out-of-sample predictions.
    rng : NumPy RandomGenerator
    size : int or tuple
        Number of samples.
    excluded : list
        Indexes of the variables to exclude when computing predictions
    """
    stacked_trees = all_trees
    if isinstance(X, Variable):
        X = X.eval()

    if size is None:
        size = ()
    elif isinstance(size, int):
        size = [size]

    flatten_size = 1
    for s in size:
        flatten_size *= s

    idx = rng.integers(0, len(stacked_trees), size=flatten_size)
    shape = stacked_trees[0][0].predict(X[0]).size

    pred = np.zeros((flatten_size, X.shape[0], shape))

    for ind, p in enumerate(pred):
        for tree in stacked_trees[idx[ind]]:
            p += np.vstack([tree.predict(x, excluded) for x in X])
    pred.reshape((*size, shape, -1))
    return pred


def plot_convergence(idata, var_name=None, kind="ecdf", figsize=None, ax=None):
    """
    Plot convergence diagnostics.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object containing the posterior samples.
    var_name : str
        Name of the BART variable to plot. Defaults to None.
    kind : str
        Type of plot to display. Options are "ecdf" (default) and "kde".
    figsize : tuple
        Figure size. Defaults to None.
    ax : matplotlib axes
        Axes on which to plot. Defaults to None.

    Returns
    -------
    ax : matplotlib axes
    """
    ess_threshold = idata.posterior.chain.size * 100
    ess = np.atleast_2d(az.ess(idata, method="bulk", var_names=var_name)[var_name].values)
    rhat = np.atleast_2d(az.rhat(idata, var_names=var_name)[var_name].values)

    if figsize is None:
        figsize = (10, 3)

    if kind == "ecdf":
        kind_func = az.plot_ecdf
        sharey = True
    elif kind == "kde":
        kind_func = az.plot_kde
        sharey = False

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=figsize, sharex="col", sharey=sharey)

    for idx, (essi, rhati) in enumerate(zip(ess, rhat)):
        kind_func(essi, ax=ax[0], plot_kwargs={"color": f"C{idx}"})
        kind_func(rhati, ax=ax[1], plot_kwargs={"color": f"C{idx}"})

    ax[0].axvline(ess_threshold, color="0.7", ls="--")
    # Assume Rhats are N(1, 0.005) iid. Then compute the 0.99 quantile
    # scaled by the sample size and use it as a threshold.
    ax[1].axvline(norm(1, 0.005).ppf(0.99 ** (1 / ess.size)), color="0.7", ls="--")

    ax[0].set_xlabel("ESS")
    ax[1].set_xlabel("R-hat")
    if kind == "kde":
        ax[0].set_yticks([])
        ax[1].set_yticks([])

    return ax


def plot_dependence(
    bartrv,
    X,
    Y=None,
    kind="pdp",
    xs_interval="linear",
    xs_values=None,
    var_idx=None,
    var_discrete=None,
    func=None,
    samples=50,
    instances=10,
    random_seed=None,
    sharey=True,
    rug=True,
    smooth=True,
    indices=None,
    grid="long",
    color="C0",
    color_mean="C0",
    alpha=0.1,
    figsize=None,
    smooth_kwargs=None,
    ax=None,
):
    """
    Partial dependence or individual conditional expectation plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : array-like
        The covariate matrix.
    Y : array-like
        The response vector.
    kind : str
        Whether to plor a partial dependence plot ("pdp") or an individual conditional expectation
        plot ("ice"). Defaults to pdp.
    xs_interval : str
        Method used to compute the values X used to evaluate the predicted function. "linear",
        evenly spaced values in the range of X. "quantiles", the evaluation is done at the specified
        quantiles of X. "insample", the evaluation is done at the values of X.
        For discrete variables these options are ommited.
    xs_values : int or list
        Values of X used to evaluate the predicted function. If ``xs_interval="linear"`` number of
        points in the evenly spaced grid. If ``xs_interval="quantiles"``quantile or sequence of
        quantiles to compute, which must be between 0 and 1 inclusive.
        Ignored when ``xs_interval="insample"``.
    var_idx : list
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : list
        List of the indices of the covariate treated as discrete.
    func : function
        Arbitrary function to apply to the predictions. Defaults to the identity function.
    samples : int
        Number of posterior samples used in the predictions. Defaults to 50
    instances : int
        Number of instances of X to plot. Only relevant if ice ``kind="ice"`` plots.
    random_seed : int
        Seed used to sample from the posterior. Defaults to None.
    sharey : bool
        Controls sharing of properties among y-axes. Defaults to True.
    rug : bool
        Whether to include a rugplot. Defaults to True.
    smooth : bool
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
        Defaults to True.
    grid : str or tuple
        How to arrange the subplots. Defaults to "long", one subplot below the other.
        Other options are "wide", one subplot next to eachother or a tuple indicating the number of
        rows and columns.
    color : matplotlib valid color
        Color used to plot the pdp or ice. Defaults to "C0"
    color_mean : matplotlib valid color
        Color used to plot the mean pdp or ice. Defaults to "C0",
    alpha : float
        Transparency level, should in the interval [0, 1].
    figsize : tuple
        Figure size. If None it will be defined automatically.
    smooth_kwargs : dict
        Additional keywords modifying the Savitzky-Golay filter.
        See scipy.signal.savgol_filter() for details.
    ax : axes
        Matplotlib axes.

    Returns
    -------
    axes: matplotlib axes
    """
    if kind not in ["pdp", "ice"]:
        raise ValueError(f"kind={kind} is not suported. Available option are 'pdp' or 'ice'")

    if xs_interval not in ["insample", "linear", "quantiles"]:
        raise ValueError(
            f"""{xs_interval} is not suported.
                          Available option are 'insample', 'linear' or 'quantiles'"""
        )

    if isinstance(X, Variable):
        X = X.eval()

    if hasattr(X, "columns") and hasattr(X, "values"):
        x_names = list(X.columns)
        X = X.values
    else:
        x_names = []

    if hasattr(Y, "name"):
        y_label = f"Predicted {Y.name}"
    else:
        y_label = "Predicted Y"

    rng = np.random.default_rng(random_seed)

    num_covariates = X.shape[1]

    indices = list(range(num_covariates))

    if var_idx is None:
        var_idx = indices
    if var_discrete is None:
        var_discrete = []

    if x_names:
        x_labels = [x_names[idx] for idx in var_idx]
    else:
        x_labels = [f"X_{idx}" for idx in var_idx]

    if xs_interval == "linear" and xs_values is None:
        xs_values = 10

    if xs_interval == "quantiles" and xs_values is None:
        xs_values = [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]

    if kind == "ice":
        instances = rng.choice(range(X.shape[0]), replace=False, size=instances)

    new_y = []
    new_x_target = []
    y_mins = []

    new_X = np.zeros_like(X)
    idx_s = list(range(X.shape[0]))
    all_trees = bartrv.owner.op.all_trees
    for i in var_idx:
        indices_mi = indices[:]
        indices_mi.pop(i)
        y_pred = []
        if kind == "pdp":
            if i in var_discrete:
                new_x_i = np.unique(X[:, i])
            else:
                if xs_interval == "linear":
                    new_x_i = np.linspace(np.nanmin(X[:, i]), np.nanmax(X[:, i]), xs_values)
                elif xs_interval == "quantiles":
                    new_x_i = np.quantile(X[:, i], q=xs_values)
                elif xs_interval == "insample":
                    new_x_i = X[:, i]

            for x_i in new_x_i:
                new_X[:, indices_mi] = X[:, indices_mi]
                new_X[:, i] = x_i
                y_pred.append(
                    np.mean(_sample_posterior(all_trees, X=new_X, rng=rng, size=samples), 1)
                )
            new_x_target.append(new_x_i)
        else:
            for instance in instances:
                new_X = X[idx_s]
                new_X[:, indices_mi] = X[:, indices_mi][instance]
                y_pred.append(
                    np.mean(_sample_posterior(all_trees, X=new_X, rng=rng, size=samples), 0)
                )
            new_x_target.append(new_X[:, i])
        y_mins.append(np.min(y_pred))
        new_y.append(np.array(y_pred).T)

    if func is not None:
        new_y = [func(nyi) for nyi in new_y]
    shape = 1
    new_y = np.array(new_y)
    if new_y[0].ndim == 3:
        shape = new_y[0].shape[0]
    if ax is None:
        if grid == "long":
            fig, axes = plt.subplots(len(var_idx) * shape, sharey=sharey, figsize=figsize)
        elif grid == "wide":
            fig, axes = plt.subplots(1, len(var_idx) * shape, sharey=sharey, figsize=figsize)
        elif isinstance(grid, tuple):
            fig, axes = plt.subplots(grid[0], grid[1], sharey=sharey, figsize=figsize)
            grid_size = grid[0] * grid[1]
            n_plots = new_y.squeeze().shape[0]
            if n_plots > grid_size:
                warnings.warn("The grid is smaller than the number of available variables to plot")
            elif n_plots < grid_size:
                for i in range(n_plots, grid[0] * grid[1]):
                    fig.delaxes(axes.flatten()[i])
                axes = axes.flatten()[:n_plots]
        axes = np.ravel(axes)
    else:
        axes = [ax]
        fig = ax.get_figure()

    x_idx = 0
    y_idx = 0
    for ax in axes:  # pylint: disable=redefined-argument-from-local
        nyi = new_y[x_idx][y_idx]
        nxi = new_x_target[x_idx]
        var = var_idx[x_idx]

        ax.set_xlabel(x_labels[x_idx])
        x_idx += 1
        if x_idx == len(var_idx):
            x_idx = 0
            y_idx += 1

        if var in var_discrete:
            if kind == "pdp":
                y_means = nyi.mean(0)
                hdi = az.hdi(nyi)
                ax.errorbar(
                    nxi,
                    y_means,
                    (y_means - hdi[:, 0], hdi[:, 1] - y_means),
                    fmt=".",
                    color=color,
                )
            else:
                ax.plot(nxi, nyi, ".", color=color, alpha=alpha)
                ax.plot(nxi, nyi.mean(1), "o", color=color_mean)
            ax.set_xticks(nxi)
        elif smooth:
            if smooth_kwargs is None:
                smooth_kwargs = {}
            smooth_kwargs.setdefault("window_length", 55)
            smooth_kwargs.setdefault("polyorder", 2)
            x_data = np.linspace(np.nanmin(nxi), np.nanmax(nxi), 200)
            x_data[0] = (x_data[0] + x_data[1]) / 2
            if kind == "pdp":
                interp = griddata(nxi, nyi.mean(0), x_data)
            else:
                interp = griddata(nxi, nyi, x_data)

            y_data = savgol_filter(interp, axis=0, **smooth_kwargs)

            if kind == "pdp":
                az.plot_hdi(nxi, nyi, color=color, fill_kwargs={"alpha": alpha}, ax=ax)
                ax.plot(x_data, y_data, color=color_mean)
            else:
                ax.plot(x_data, y_data.mean(1), color=color_mean)
                ax.plot(x_data, y_data, color=color, alpha=alpha)

        else:
            idx = np.argsort(nxi)
            if kind == "pdp":
                az.plot_hdi(
                    nxi,
                    nyi,
                    smooth=smooth,
                    fill_kwargs={"alpha": alpha, "color": color},
                    ax=ax,
                )
                ax.plot(nxi[idx], nyi[idx].mean(0), color=color)
            else:
                ax.plot(nxi[idx], nyi[idx], color=color, alpha=alpha)
                ax.plot(nxi[idx], nyi[idx].mean(1), color=color_mean)

        if rug:
            lower = np.min(y_mins)
            ax.plot(X[:, var], np.full_like(X[:, var], lower), "k|")

    fig.text(-0.05, 0.5, y_label, va="center", rotation="vertical", fontsize=15)
    return axes


def plot_variable_importance(
    idata, bartrv, X, labels=None, sort_vars=True, figsize=None, samples=100, random_seed=None
):
    """
    Estimates variable importance from the BART-posterior.

    Parameters
    ----------
    idata: InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : array-like
        The covariate matrix.
    labels : list
        List of the names of the covariates. If X is a DataFrame the names of the covariables will
        be taken from it and this argument will be ignored.
    sort_vars : bool
        Whether to sort the variables according to their variable importance. Defaults to True.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    samples : int
        Number of predictions used to compute correlation for subsets of variables. Defaults to 100
    random_seed : int
        random_seed used to sample from the posterior. Defaults to None.
    Returns
    -------
    idxs: indexes of the covariates from higher to lower relative importance
    axes: matplotlib axes
    """
    _, axes = plt.subplots(2, 1, figsize=figsize)

    if hasattr(X, "columns") and hasattr(X, "values"):
        labels = X.columns
        X = X.values

    var_imp = idata.sample_stats["variable_inclusion"].mean(("chain", "draw")).values
    if labels is None:
        labels = np.arange(len(var_imp))
    else:
        labels = np.array(labels)

    rng = np.random.default_rng(random_seed)

    ticks = np.arange(len(var_imp), dtype=int)
    idxs = np.argsort(var_imp)
    subsets = [idxs[:-i] for i in range(1, len(idxs))]
    subsets.append(None)

    if sort_vars:
        indices = idxs[::-1]
    else:
        indices = np.arange(len(var_imp))
    axes[0].plot((var_imp / var_imp.sum())[indices], "o-")
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(labels[indices])
    axes[0].set_xlabel("covariables")
    axes[0].set_ylabel("importance")

    all_trees = bartrv.owner.op.all_trees

    predicted_all = _sample_posterior(all_trees, X=X, rng=rng, size=samples, excluded=None)

    ev_mean = np.zeros(len(var_imp))
    ev_hdi = np.zeros((len(var_imp), 2))
    for idx, subset in enumerate(subsets):
        predicted_subset = _sample_posterior(all_trees, X=X, rng=rng, size=samples, excluded=subset)
        pearson = np.zeros(samples)
        for j in range(samples):
            pearson[j] = (
                pearsonr(predicted_all[j].flatten(), predicted_subset[j].flatten())[0]
            ) ** 2
        ev_mean[idx] = np.mean(pearson)
        ev_hdi[idx] = az.hdi(pearson)

    axes[1].errorbar(ticks, ev_mean, np.array((ev_mean - ev_hdi[:, 0], ev_hdi[:, 1] - ev_mean)))

    axes[1].axhline(ev_mean[-1], ls="--", color="0.5")
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(ticks + 1)
    axes[1].set_xlabel("number of covariables")
    axes[1].set_ylabel("R²", rotation=0, labelpad=12)
    axes[1].set_ylim(0, 1)

    axes[0].set_xlim(-0.5, len(var_imp) - 0.5)
    axes[1].set_xlim(-0.5, len(var_imp) - 0.5)

    return idxs[::-1], axes
