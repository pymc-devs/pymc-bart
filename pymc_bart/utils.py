# pylint: disable=too-many-branches
"""Utility function for variable selection and bart interpretability."""

import warnings
from typing import Any, Callable, Optional, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from numba import jit
from pytensor.tensor.variable import Variable
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.stats import norm

from .tree import Tree

TensorLike = Union[npt.NDArray, pt.TensorVariable]


def _sample_posterior(
    all_trees: list[list[Tree]],
    X: TensorLike,
    rng: np.random.Generator,
    size: Optional[Union[int, tuple[int, ...]]] = None,
    excluded: Optional[list[int]] = None,
    shape: int = 1,
) -> npt.NDArray:
    """
    Generate samples from the BART-posterior.

    Parameters
    ----------
    all_trees : list
        List of all trees sampled from a posterior
    X : tensor-like
        A covariate matrix. Use the same used to fit BART for in-sample predictions or a new one for
        out-of-sample predictions.
    rng : NumPy RandomGenerator
    size : int or tuple
        Number of samples.
    excluded : Optional[npt.NDArray[np.int_]]
        Indexes of the variables to exclude when computing predictions
    """
    stacked_trees = all_trees

    if isinstance(X, Variable):
        X = X.eval()

    if size is None:
        size_iter: Union[list, tuple] = (1,)
    elif isinstance(size, int):
        size_iter = [size]
    else:
        size_iter = size

    flatten_size = 1
    for s in size_iter:
        flatten_size *= s

    idx = rng.integers(0, len(stacked_trees), size=flatten_size)

    trees_shape = len(stacked_trees[0])
    leaves_shape = shape // trees_shape

    pred = np.zeros((flatten_size, trees_shape, leaves_shape, X.shape[0]))

    for ind, p in enumerate(pred):
        for odim, odim_trees in enumerate(stacked_trees[idx[ind]]):
            for tree in odim_trees:
                p[odim] += tree.predict(x=X, excluded=excluded, shape=leaves_shape)

    return pred.transpose((0, 3, 1, 2)).reshape((*size_iter, -1, shape))


def plot_convergence(
    idata: az.InferenceData,
    var_name: Optional[str] = None,
    kind: str = "ecdf",
    figsize: Optional[tuple[float, float]] = None,
    ax=None,
) -> list[plt.Axes]:
    """
    Plot convergence diagnostics.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object containing the posterior samples.
    var_name : Optional[str]
        Name of the BART variable to plot. Defaults to None.
    kind : str
        Type of plot to display. Options are "ecdf" (default) and "kde".
    figsize : Optional[tuple[float, float]], by default None.
        Figure size. Defaults to None.
    ax : matplotlib axes
        Axes on which to plot. Defaults to None.

    Returns
    -------
    list[ax] : matplotlib axes
    """
    ess_threshold = idata["posterior"]["chain"].size * 100
    ess = np.atleast_2d(az.ess(idata, method="bulk", var_names=var_name)[var_name].values)
    rhat = np.atleast_2d(az.rhat(idata, var_names=var_name)[var_name].values)

    if figsize is None:
        figsize = (10, 3)

    if kind == "ecdf":
        kind_func: Callable[..., Any] = az.plot_ecdf
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


def plot_ice(
    bartrv: Variable,
    X: npt.NDArray,
    Y: Optional[npt.NDArray] = None,
    var_idx: Optional[list[int]] = None,
    var_discrete: Optional[list[int]] = None,
    func: Optional[Callable] = None,
    centered: Optional[bool] = True,
    samples: int = 100,
    instances: int = 30,
    random_seed: Optional[int] = None,
    sharey: bool = True,
    smooth: bool = True,
    grid: str = "long",
    color="C0",
    color_mean: str = "C0",
    alpha: float = 0.1,
    figsize: Optional[tuple[float, float]] = None,
    smooth_kwargs: Optional[dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
) -> list[plt.Axes]:
    """
    Individual conditional expectation plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray
        The covariate matrix.
    Y : Optional[npt.NDArray], by default None.
        The response vector.
    var_idx : Optional[list[int]], by default None.
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : Optional[list[int]], by default None.
        List of the indices of the covariate treated as discrete.
    func : Optional[Callable], by default None.
        Arbitrary function to apply to the predictions. Defaults to the identity function.
    centered : bool
        If True the result is centered around the partial response evaluated at the lowest value in
        ``xs_interval``. Defaults to True.
    samples : int
        Number of posterior samples used in the predictions. Defaults to 100
    instances : int
        Number of instances of X to plot. Defaults to 30.
    random_seed : Optional[int], by default None.
        Seed used to sample from the posterior. Defaults to None.
    sharey : bool
        Controls sharing of properties among y-axes. Defaults to True.
    smooth : bool
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
        Defaults to True.
    grid : str or tuple
        How to arrange the subplots. Defaults to "long", one subplot below the other.
        Other options are "wide", one subplot next to each other or a tuple indicating the number of
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
    all_trees = bartrv.owner.op.all_trees
    rng = np.random.default_rng(random_seed)

    if func is None:

        def identity(x):
            return x

        func = identity

    (
        X,
        x_labels,
        y_label,
        indices,
        var_idx,
        var_discrete,
        _,
        _,
    ) = _prepare_plot_data(X, Y, "linear", None, var_idx, var_discrete)

    fig, axes, shape = _create_figure_axes(bartrv, var_idx, grid, sharey, figsize, ax)

    instances_ary = rng.choice(range(X.shape[0]), replace=False, size=instances)
    idx_s = list(range(X.shape[0]))

    count = 0
    for i_var, var in enumerate(var_idx):
        indices_mi = indices[:]
        indices_mi.remove(var)
        y_pred = []
        for instance in instances_ary:
            fake_X = X[idx_s]
            fake_X[:, indices_mi] = X[:, indices_mi][instance]
            y_pred.append(
                np.mean(
                    _sample_posterior(all_trees, X=fake_X, rng=rng, size=samples, shape=shape),
                    0,
                )
            )

        new_x = fake_X[:, var]
        p_d = func(np.array(y_pred))

        for s_i in range(shape):
            if centered:
                p_di = p_d[:, :, s_i] - p_d[:, :, s_i][:, 0][:, None]
            else:
                p_di = p_d[:, :, s_i]
            if var in var_discrete:
                axes[count].plot(new_x, p_di.mean(0), "o", color=color_mean)
                axes[count].plot(new_x, p_di.T, ".", color=color, alpha=alpha)
            elif smooth:
                x_data, y_data = _smooth_mean(new_x, p_di, "ice", smooth_kwargs)
                axes[count].plot(x_data, y_data.mean(1), color=color_mean)
                axes[count].plot(x_data, y_data, color=color, alpha=alpha)
            else:
                idx = np.argsort(new_x)
                axes[count].plot(new_x[idx], p_di.mean(0)[idx], color=color_mean)
                axes[count].plot(new_x[idx], p_di.T[idx], color=color, alpha=alpha)
            axes[count].set_xlabel(x_labels[i_var])

            count += 1

    fig.text(-0.05, 0.5, y_label, va="center", rotation="vertical", fontsize=15)

    return axes


def plot_pdp(
    bartrv: Variable,
    X: npt.NDArray,
    Y: Optional[npt.NDArray] = None,
    xs_interval: str = "quantiles",
    xs_values: Optional[Union[int, list[float]]] = None,
    var_idx: Optional[list[int]] = None,
    var_discrete: Optional[list[int]] = None,
    func: Optional[Callable] = None,
    samples: int = 200,
    ref_line: bool = True,
    random_seed: Optional[int] = None,
    sharey: bool = True,
    smooth: bool = True,
    grid: str = "long",
    color="C0",
    color_mean: str = "C0",
    alpha: float = 0.1,
    figsize: Optional[tuple[float, float]] = None,
    smooth_kwargs: Optional[dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
) -> list[plt.Axes]:
    """
    Partial dependence plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray
        The covariate matrix.
    Y : Optional[npt.NDArray], by default None.
        The response vector.
    xs_interval : str
        Method used to compute the values X used to evaluate the predicted function. "linear",
        evenly spaced values in the range of X. "quantiles", the evaluation is done at the specified
        quantiles of X. "insample", the evaluation is done at the values of X.
        For discrete variables these options are ommited.
    xs_values : Optional[Union[int, list[float]]], by default None.
        Values of X used to evaluate the predicted function. If ``xs_interval="linear"`` number of
        points in the evenly spaced grid. If ``xs_interval="quantiles"`` quantile or sequence of
        quantiles to compute, which must be between 0 and 1 inclusive.
        Ignored when ``xs_interval="insample"``.
    var_idx : Optional[list[int]], by default None.
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : Optional[list[int]], by default None.
        List of the indices of the covariate treated as discrete.
    func : Optional[Callable], by default None.
        Arbitrary function to apply to the predictions. Defaults to the identity function.
    samples : int
        Number of posterior samples used in the predictions. Defaults to 200
    ref_line : bool
        If True a reference line is plotted at the mean of the partial dependence. Defaults to True.
    random_seed : Optional[int], by default None.
        Seed used to sample from the posterior. Defaults to None.
    sharey : bool
        Controls sharing of properties among y-axes. Defaults to True.
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
    all_trees: list = bartrv.owner.op.all_trees
    rng = np.random.default_rng(random_seed)

    if func is None:

        def identity(x):
            return x

        func = identity

    (
        X,
        x_labels,
        y_label,
        indices,
        var_idx,
        var_discrete,
        xs_interval,
        xs_values,
    ) = _prepare_plot_data(X, Y, xs_interval, xs_values, var_idx, var_discrete)

    fig, axes, shape = _create_figure_axes(bartrv, var_idx, grid, sharey, figsize, ax)

    count = 0
    fake_X = _create_pdp_data(X, xs_interval, xs_values)
    null_pd = []
    for var in range(len(var_idx)):
        excluded = indices[:]
        excluded.remove(var)
        p_d = func(
            _sample_posterior(
                all_trees, X=fake_X, rng=rng, size=samples, excluded=excluded, shape=shape
            )
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="hdi currently interprets 2d data")
            new_x = fake_X[:, var]
            for s_i in range(shape):
                p_di = p_d[:, :, s_i]
                null_pd.append(p_di.mean())
                if var in var_discrete:
                    _, idx_uni = np.unique(new_x, return_index=True)
                    y_means = p_di.mean(0)[idx_uni]
                    hdi = az.hdi(p_di)[idx_uni]
                    axes[count].errorbar(
                        new_x[idx_uni],
                        y_means,
                        (y_means - hdi[:, 0], hdi[:, 1] - y_means),
                        fmt=".",
                        color=color,
                    )
                    axes[count].set_xticks(new_x[idx_uni])
                else:
                    az.plot_hdi(
                        new_x,
                        p_di,
                        smooth=smooth,
                        fill_kwargs={"alpha": alpha, "color": color},
                        ax=axes[count],
                    )
                    if smooth:
                        x_data, y_data = _smooth_mean(new_x, p_di, "pdp", smooth_kwargs)
                        axes[count].plot(x_data, y_data, color=color_mean)
                    else:
                        axes[count].plot(new_x, p_di.mean(0), color=color_mean)
                axes[count].set_xlabel(x_labels[var])

                count += 1

    if ref_line:
        ref_val = sum(null_pd) / len(null_pd)
        for ax_ in np.ravel(axes):
            ax_.axhline(ref_val, color="0.7", linestyle="--")

    fig.text(-0.05, 0.5, y_label, va="center", rotation="vertical", fontsize=15)

    return axes


def _create_figure_axes(
    bartrv: Variable,
    var_idx: list[int],
    grid: str = "long",
    sharey: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, list[plt.Axes], int]:
    """
    Create and return the figure and axes objects for plotting the variables.

    Partial dependence plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    var_idx : Optional[list[int]], by default None.
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : Optional[list[int]], by default None.
    grid : str or tuple
        How to arrange the subplots. Defaults to "long", one subplot below the other.
        Other options are "wide", one subplot next to each other or a tuple indicating the number of
        rows and columns.
    sharey : bool
        Controls sharing of properties among y-axes. Defaults to True.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    ax : axes
        Matplotlib axes.


    Returns
    -------
    tuple[plt.Figure, list[plt.Axes], int]
        A tuple containing the figure object, list of axes objects, and the shape value.
    """
    if bartrv.ndim == 1:  # type: ignore
        shape = 1
    else:
        shape = bartrv.eval().shape[0]

    n_plots = len(var_idx) * shape

    if ax is None:
        fig, axes = _get_axes(grid, n_plots, False, sharey, figsize)

    elif isinstance(ax, np.ndarray):
        axes = ax
        fig = ax[0].get_figure()
    else:
        axes = [ax]
        fig = ax.get_figure()  # type: ignore

    return fig, axes, shape


def _get_axes(grid, n_plots, sharex, sharey, figsize):
    if grid == "long":
        fig, axes = plt.subplots(n_plots, sharex=sharex, sharey=sharey, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
    elif grid == "wide":
        fig, axes = plt.subplots(1, n_plots, sharex=sharex, sharey=sharey, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
    elif isinstance(grid, tuple):
        grid_size = grid[0] * grid[1]
        if n_plots > grid_size:
            warnings.warn(
                """The grid is smaller than the number of available variables to plot.
                Automatically adjusting the grid size."""
            )
            grid = (n_plots // grid[1] + (n_plots % grid[1] > 0), grid[1])

        fig, axes = plt.subplots(*grid, sharey=sharey, figsize=figsize)
        axes = np.ravel(axes)

        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])
        axes = axes[:n_plots]
    return fig, axes


def _prepare_plot_data(
    X: npt.NDArray,
    Y: Optional[npt.NDArray] = None,
    xs_interval: str = "quantiles",
    xs_values: Optional[Union[int, list[float]]] = None,
    var_idx: Optional[list[int]] = None,
    var_discrete: Optional[list[int]] = None,
) -> tuple[
    npt.NDArray,
    list[str],
    str,
    list[int],
    list[int],
    list[int],
    str,
    Union[int, None, list[float]],
]:
    """
    Prepare data for plotting.

    Parameters
    ----------
    X : PyTensor Variable, Pandas DataFrame, Polars DataFrame or Numpy array
        Input data.
    Y : array-like
        Target data.
    xs_interval : str
        Interval for x-axis. Available options are 'insample', 'linear' or 'quantiles'.
    xs_values : int or list
        Number of points for 'linear' or list of quantiles for 'quantiles'.
    var_idx : None or list
        Indices of variables to plot.
    var_discrete : None or list
        Indices of discrete variables.

    Returns
    -------
    X : Numpy array
        Input data.
    x_labels : list
        Names of variables.
    y_label : str
        Name of target variable.
    var_idx: list
        Indices of variables to plot.
    var_discrete : list
        Indices of discrete variables.
    xs_interval : str
        Interval for x-axis.
    xs_values : int or list
        Number of points for 'linear' or list of quantiles for 'quantiles'.
    """
    if xs_interval not in ["insample", "linear", "quantiles"]:
        raise ValueError(
            f"""{xs_interval} is not suported.
                        Available option are 'insample', 'linear' or 'quantiles'"""
        )

    if isinstance(X, Variable):
        X = X.eval()

    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        x_names = list(X.columns)
        X = X.to_numpy()
    else:
        x_names = []

    if Y is not None and hasattr(Y, "name"):
        y_label = f"Partial {Y.name}"
    else:
        y_label = "Partial Y"

    indices = list(range(X.shape[1]))

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

    return X, x_labels, y_label, indices, var_idx, var_discrete, xs_interval, xs_values


def _create_pdp_data(
    X: npt.NDArray,
    xs_interval: str,
    xs_values: Optional[Union[int, list[float]]] = None,
) -> npt.NDArray:
    """
    Create data for partial dependence plot.

    Parameters
    ----------
    X : Numpy array
        Input data.
    xs_interval : str
        Interval for x-axis. Available options are 'insample', 'linear' or 'quantiles'.
    xs_values : int or list
        Number of points for 'linear' or list of quantiles for 'quantiles'.

    Returns
    -------
    npt.NDArray
        A 2D array for the fake_X data.
    """
    if xs_interval == "insample":
        return X
    else:
        if xs_interval == "linear" and isinstance(xs_values, int):
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            fake_X = np.linspace(min_vals, max_vals, num=xs_values, axis=0)
        elif xs_interval == "quantiles" and isinstance(xs_values, list):
            fake_X = np.quantile(X, q=xs_values, axis=0)

        return fake_X


def _smooth_mean(
    new_x: npt.NDArray,
    p_di: npt.NDArray,
    kind: str = "pdp",
    smooth_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth the mean data for plotting.

    Parameters
    ----------
    new_x : np.ndarray
        The x-axis data.
    p_di : np.ndarray
        The distribution of partial dependence from which to comptue the smoothed mean.
    kind : str, optional
        The type of plot. Possible values are "pdp" or "ice".
    smooth_kwargs : Optional[dict[str, Any]], optional
        Additional keyword arguments for the smoothing function. Defaults to None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing a grid for the x-axis data and the corresponding smoothed y-axis data.

    """
    if smooth_kwargs is None:
        smooth_kwargs = {}
    smooth_kwargs.setdefault("window_length", 55)
    smooth_kwargs.setdefault("polyorder", 2)
    x_data = np.linspace(np.nanmin(new_x), np.nanmax(new_x), 200)
    x_data[0] = (x_data[0] + x_data[1]) / 2
    if kind == "pdp":
        interp = griddata(new_x, p_di.mean(0), x_data)
    else:
        interp = griddata(new_x, p_di.T, x_data)
    y_data = savgol_filter(interp, axis=0, **smooth_kwargs)
    return x_data, y_data


def get_variable_inclusion(idata, X, labels=None, to_kulprit=False):
    """
    Get the normalized variable inclusion from BART model.

    Parameters
    ----------
    idata : InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    X : npt.NDArray
        The covariate matrix.
    labels : Optional[list[str]]
        List of the names of the covariates. If X is a DataFrame the names of the covariables will
        be taken from it and this argument will be ignored.
    to_kulprit : bool
        If True, the function will return a list of list with the variables names.
        This list can be passed as a path to Kulprit's project method. Defaults to False.
    Returns
    -------
    VI_norm : npt.NDArray
        Normalized variable inclusion.
    labels : list[str]
        List of the names of the covariates.
    """
    VIs = idata["sample_stats"]["variable_inclusion"].mean(("chain", "draw")).values
    VI_norm = VIs / VIs.sum()
    idxs = np.argsort(VI_norm)

    indices = idxs[::-1]
    n_vars = len(indices)

    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        labels = X.columns

    if labels is None:
        labels = np.arange(n_vars).astype(str)

    label_list = labels.to_list()

    if to_kulprit:
        return [label_list[:idx] for idx in range(n_vars)]
    else:
        return VI_norm[indices], label_list


def plot_variable_inclusion(idata, X, labels=None, figsize=None, plot_kwargs=None, ax=None):
    """
    Plot normalized variable inclusion from BART model.

    Parameters
    ----------
    idata : InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    X : npt.NDArray
        The covariate matrix.
    labels : Optional[list[str]]
        List of the names of the covariates. If X is a DataFrame the names of the covariables will
        be taken from it and this argument will be ignored.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    plot_kwargs : dict
        Additional keyword arguments for the plot. Defaults to None.
        Valid keys are:
        - color: matplotlib valid color for VI
        - marker: matplotlib valid marker for VI
        - ls: matplotlib valid linestyle for the VI line
        - rotation: float, rotation of the x-axis labels
    ax : axes
        Matplotlib axes.

    Returns
    -------
    axes: matplotlib axes
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    VI_norm, labels = get_variable_inclusion(idata, X, labels)
    n_vars = len(labels)

    new_labels = ["+ " + ele if index != 0 else ele for index, ele in enumerate(labels)]

    ticks = np.arange(n_vars, dtype=int)

    if figsize is None:
        figsize = (8, 3)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.axhline(1 / n_vars, color="0.5", linestyle="--")
    ax.plot(
        VI_norm,
        color=plot_kwargs.get("color", "k"),
        marker=plot_kwargs.get("marker", "o"),
        ls=plot_kwargs.get("ls", "-"),
    )

    ax.set_xticks(ticks, new_labels, rotation=plot_kwargs.get("rotation", 0))
    ax.set_ylim(0, 1)

    return ax


def compute_variable_importance(  # noqa: PLR0915 PLR0912
    idata: az.InferenceData,
    bartrv: Variable,
    X: npt.NDArray,
    method: str = "VI",
    fixed: int = 0,
    samples: int = 50,
    random_seed: Optional[int] = None,
) -> dict[str, object]:
    """
    Estimates variable importance from the BART-posterior.

    Parameters
    ----------
    idata : InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray
        The covariate matrix.
    method : str
        Method used to rank variables. Available options are "VI" (default), "backward"
        and "backward_VI".
        The R squared will be computed following this ranking.
        "VI" counts how many times each variable is included in the posterior distribution
        of trees. "backward" uses a backward search based on the R squared.
        "backward_VI" combines both methods with the backward search excluding
        the ``fixed`` number of variables with the lowest variable inclusion.
        "VI" is the fastest method, while "backward" is the slowest.
    fixed : Optional[int]
        Number of variables to fix in the backward search. Defaults to None.
        Must be greater than 0 and less than the number of variables.
        Ignored if method is "VI" or "backward".
    samples : int
        Number of predictions used to compute correlation for subsets of variables. Defaults to 50
    random_seed : Optional[int]
        random_seed used to sample from the posterior. Defaults to None.

    Returns
    -------
    vi_results: dictionary
    """
    if method not in ["VI", "backward", "backward_VI"]:
        raise ValueError("method must be 'VI', 'backward' or 'backward_VI'")

    rng = np.random.default_rng(random_seed)

    all_trees = bartrv.owner.op.all_trees

    if bartrv.ndim == 1:  # type: ignore
        shape = 1
    else:
        shape = bartrv.eval().shape[0]

    n_vars = X.shape[1]

    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        labels = X.columns
        X = X.to_numpy()
    else:
        labels = np.arange(n_vars).astype(str)

    r2_mean: npt.NDArray = np.zeros(n_vars)
    r2_hdi: npt.NDArray = np.zeros((n_vars, 2))
    preds: npt.NDArray = np.zeros((n_vars, samples, *bartrv.eval().T.shape))

    if method == "backward_VI":
        if fixed >= n_vars:
            raise ValueError("fixed must be less than the number of variables")
        elif fixed < 1:
            raise ValueError("fixed must be greater than 0")
        init = fixed + 1
    else:
        fixed = 0
        init = 0

    predicted_all = _sample_posterior(
        all_trees, X=X, rng=rng, size=samples, excluded=None, shape=shape
    )

    if method in ["VI", "backward_VI"]:
        idxs = np.argsort(
            idata["sample_stats"]["variable_inclusion"].mean(("chain", "draw")).values
        )
        subsets: list[list[int]] = [list(idxs[:-i]) for i in range(1, len(idxs))]
        subsets.append(None)  # type: ignore

        if method == "backward_VI":
            subsets = subsets[-init:]

        indices: list[int] = list(idxs[::-1])

        for idx, subset in enumerate(subsets):
            predicted_subset = _sample_posterior(
                all_trees=all_trees,
                X=X,
                rng=rng,
                size=samples,
                excluded=subset,
                shape=shape,
            )
            r_2 = np.array(
                [pearsonr2(predicted_all[j], predicted_subset[j]) for j in range(samples)]
            )
            r2_mean[idx] = np.mean(r_2)
            r2_hdi[idx] = az.hdi(r_2)
            preds[idx] = predicted_subset.squeeze()

    if method in ["backward", "backward_VI"]:
        if method == "backward_VI":
            least_important_vars: list[int] = indices[-fixed:]
            r2_mean_vi = r2_mean[:init]
            r2_hdi_vi = r2_hdi[:init]
            preds_vi = preds[:init]
            r2_mean = np.zeros(n_vars - fixed - 1)
            r2_hdi = np.zeros((n_vars - fixed - 1, 2))
            preds = np.zeros((n_vars - fixed - 1, samples, bartrv.eval().shape[0]))
        else:
            least_important_vars = []

        # Iterate over each variable to determine its contribution
        # least_important_vars tracks the variable with the lowest contribution
        # at the current stage. One new variable is added at each iteration.
        for i_var in range(init, n_vars):
            # Generate all possible subsets by adding one variable at a time to
            # least_important_vars
            subsets = generate_sequences(n_vars, i_var, least_important_vars)
            max_r_2 = -np.inf

            # Iterate over each subset to find the one with the maximum Pearson correlation
            for subset in subsets:
                # Sample posterior predictions excluding a subset of variables
                predicted_subset = _sample_posterior(
                    all_trees=all_trees,
                    X=X,
                    rng=rng,
                    size=samples,
                    excluded=subset,
                    shape=shape,
                )
                # Calculate Pearson correlation for each sample and find the mean
                r_2 = np.zeros(samples)
                for j in range(samples):
                    r_2[j] = pearsonr2(predicted_all[j], predicted_subset[j])
                mean_r_2 = np.mean(r_2, dtype=float)
                # Identify the least important combination of variables
                # based on the maximum mean squared Pearson correlation
                if mean_r_2 > max_r_2:
                    max_r_2 = mean_r_2
                    least_important_subset = subset
                    r_2_without_least_important_vars = r_2
                    least_important_samples = predicted_subset

            # Save values for plotting later
            r2_mean[i_var - init] = max_r_2
            r2_hdi[i_var - init] = az.hdi(r_2_without_least_important_vars)
            preds[i_var - init] = least_important_samples.squeeze()

            # extend current list of least important variable
            for var_i in least_important_subset:
                if var_i not in least_important_vars:
                    least_important_vars.append(var_i)

        # Add the remaining variables to the list of least important variables
        for var_i in range(n_vars):
            if var_i not in least_important_vars:
                least_important_vars.append(var_i)

        if method == "backward_VI":
            r2_mean = np.concatenate((r2_mean[::-1], r2_mean_vi))
            r2_hdi = np.concatenate((r2_hdi[::-1], r2_hdi_vi))
            preds = np.concatenate((preds[::-1], preds_vi))
        else:
            r2_mean = r2_mean[::-1]
            r2_hdi = r2_hdi[::-1]
            preds = preds[::-1]

        indices = least_important_vars[::-1]

    labels = np.array(
        ["+ " + ele if index != 0 else ele for index, ele in enumerate(labels[indices])]
    )

    vi_results = {
        "indices": np.asarray(indices),
        "labels": labels,
        "r2_mean": r2_mean,
        "r2_hdi": r2_hdi,
        "preds": preds,
        "preds_all": predicted_all.squeeze(),
    }
    return vi_results


def plot_variable_importance(
    vi_results: dict,
    submodels: Optional[Union[list[int], np.ndarray, tuple[int, ...]]] = None,
    labels: Optional[list[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Estimates variable importance from the BART-posterior.

    Parameters
    ----------
    vi_results: Dictionary
        Dictionary computed with `compute_variable_importance`
    submodels : Optional[Union[list[int], np.ndarray]]
        List of the indices of the submodels to plot. Defaults to None, all variables are ploted.
        The indices correspond to order computed by `compute_variable_importance`.
        For example `submodels=[0,1]` will plot the two most important variables.
        `submodels=[1,0]` is equivalent as values are sorted before use.
    labels : Optional[list[str]]
        List of the names of the covariates. If X is a DataFrame the names of the covariables will
        be taken from it and this argument will be ignored.
    plot_kwargs : dict
        Additional keyword arguments for the plot. Defaults to None.
        Valid keys are:
        - color_r2: matplotlib valid color for error bars
        - marker_r2: matplotlib valid marker for the mean R squared
        - marker_fc_r2: matplotlib valid marker face color for the mean R squared
        - ls_ref: matplotlib valid linestyle for the reference line
        - color_ref: matplotlib valid color for the reference line
        - rotation: float, rotation angle of the x-axis labels. Defaults to 0.
    ax : axes
        Matplotlib axes.

    Returns
    -------
    axes: matplotlib axes
    """
    if submodels is None:
        submodels = np.sort(vi_results["indices"])
    else:
        submodels = np.sort(submodels)

    indices = vi_results["indices"][submodels]
    r2_mean = vi_results["r2_mean"][submodels]
    r2_hdi = vi_results["r2_hdi"][submodels]
    preds = vi_results["preds"][submodels]
    preds_all = vi_results["preds_all"]
    samples = preds.shape[1]

    n_vars = len(indices)
    ticks = np.arange(n_vars, dtype=int)

    if plot_kwargs is None:
        plot_kwargs = {}

    if figsize is None:
        figsize = (8, 3)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if labels is None:
        labels = vi_results["labels"][submodels]

    r_2_ref = np.array([pearsonr2(preds_all[j], preds_all[j + 1]) for j in range(samples - 1)])

    r2_yerr_min = np.clip(r2_mean - r2_hdi[:, 0], 0, None)
    r2_yerr_max = np.clip(r2_hdi[:, 1] - r2_mean, 0, None)

    ax.errorbar(
        ticks,
        r2_mean,
        np.array((r2_yerr_min, r2_yerr_max)),
        color=plot_kwargs.get("color_r2", "k"),
        fmt=plot_kwargs.get("marker_r2", "o"),
        mfc=plot_kwargs.get("marker_fc_r2", "white"),
    )
    ax.axhline(
        np.mean(r_2_ref),
        ls=plot_kwargs.get("ls_ref", "--"),
        color=plot_kwargs.get("color_ref", "grey"),
    )
    ax.fill_between(
        [-0.5, n_vars - 0.5],
        *az.hdi(r_2_ref),
        alpha=0.1,
        color=plot_kwargs.get("color_ref", "grey"),
    )
    ax.set_xticks(
        ticks,
        labels,
        rotation=plot_kwargs.get("rotation", 0),
    )
    ax.set_ylabel("R²", rotation=0, labelpad=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, n_vars - 0.5)

    return ax


def plot_scatter_submodels(
    vi_results: dict,
    func: Optional[Callable] = None,
    submodels: Optional[Union[list[int], np.ndarray]] = None,
    grid: str = "long",
    labels: Optional[list[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
) -> list[plt.Axes]:
    """
    Plot submodel's predictions against reference-model's predictions.

    Parameters
    ----------
    vi_results : Dictionary
        Dictionary computed with `compute_variable_importance`
    func : Optional[Callable], by default None.
        Arbitrary function to apply to the predictions. Defaults to the identity function.
    submodels : Optional[Union[list[int], np.ndarray]]
        List of the indices of the submodels to plot. Defaults to None, all variables are ploted.
        The indices correspond to order computed by `compute_variable_importance`.
        For example `submodels=[0,1]` will plot the two most important variables.
        `submodels=[1,0]` is equivalent as values are sorted before use.
    grid : str or tuple
        How to arrange the subplots. Defaults to "long", one subplot below the other.
        Other options are "wide", one subplot next to each other or a tuple indicating the number
        of rows and columns.
    labels : Optional[list[str]]
        List of the names of the covariates.
    plot_kwargs : dict
        Additional keyword arguments for the plot. Defaults to None.
        Valid keys are:
        - marker_scatter: matplotlib valid marker for the scatter plot
        - color_scatter: matplotlib valid color for the scatter plot
        - alpha_scatter: matplotlib valid alpha for the scatter plot
        - color_ref: matplotlib valid color for the 45 degree line
        - ls_ref: matplotlib valid linestyle for the reference line
    axes : axes
        Matplotlib axes.

    Returns
    -------
    axes: matplotlib axes
    """
    if submodels is None:
        submodels = np.sort(vi_results["indices"])
    else:
        submodels = np.sort(submodels)

    indices = vi_results["indices"][submodels]
    preds_sub = vi_results["preds"][submodels]
    preds_all = vi_results["preds_all"]

    if labels is None:
        labels = vi_results["labels"][submodels]

    # handle categorical regression case:
    n_cats = None
    if preds_all.ndim > 2:
        n_cats = preds_all.shape[-1]
        indices = np.tile(indices, n_cats)

    if ax is None:
        _, ax = _get_axes(grid, len(indices), True, True, figsize)

    if plot_kwargs is None:
        plot_kwargs = {}

    if func is not None:
        preds_sub = func(preds_sub)
        preds_all = func(preds_all)

    min_ = min(np.min(preds_sub), np.min(preds_all))
    max_ = max(np.max(preds_sub), np.max(preds_all))

    # handle categorical regression case:
    if n_cats is not None:
        i = 0
        for cat in range(n_cats):
            for pred_sub, x_label in zip(preds_sub, labels):
                ax[i].plot(
                    pred_sub[..., cat],
                    preds_all[..., cat],
                    marker=plot_kwargs.get("marker_scatter", "."),
                    ls="",
                    color=plot_kwargs.get("color_scatter", f"C{cat}"),
                    alpha=plot_kwargs.get("alpha_scatter", 0.1),
                )
                ax[i].set(xlabel=x_label, ylabel="ref model", title=f"Category {cat}")
                ax[i].axline(
                    [min_, min_],
                    [max_, max_],
                    color=plot_kwargs.get("color_ref", "0.5"),
                    ls=plot_kwargs.get("ls_ref", "--"),
                )
                i += 1
    else:
        for pred_sub, x_label, axi in zip(preds_sub, labels, ax.ravel()):
            axi.plot(
                pred_sub,
                preds_all,
                marker=plot_kwargs.get("marker_scatter", "."),
                ls="",
                color=plot_kwargs.get("color_scatter", "C0"),
                alpha=plot_kwargs.get("alpha_scatter", 0.1),
            )
            axi.set(xlabel=x_label, ylabel="ref model")
            axi.axline(
                [min_, min_],
                [max_, max_],
                color=plot_kwargs.get("color_ref", "0.5"),
                ls=plot_kwargs.get("ls_ref", "--"),
            )
    return ax


def generate_sequences(n_vars, i_var, include):
    """Generate combinations of variables"""
    if i_var:
        sequences = [tuple(include + [i]) for i in range(n_vars) if i not in include]
    else:
        sequences = [()]
    return sequences


@jit(nopython=True)
def pearsonr2(A, B):
    """Compute the squared Pearson correlation coefficient"""
    A = A.flatten()
    B = B.flatten()
    am = A - np.mean(A)
    bm = B - np.mean(B)
    return (am @ bm) ** 2 / (np.sum(am**2) * np.sum(bm**2))
