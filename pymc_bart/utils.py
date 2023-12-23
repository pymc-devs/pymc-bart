"""Utility function for variable selection and bart interpretability."""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pytensor.tensor.var import Variable
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.stats import norm, pearsonr

from .tree import Tree

TensorLike = Union[npt.NDArray[np.float_], pt.TensorVariable]


def _sample_posterior(
    all_trees: List[List[Tree]],
    X: TensorLike,
    rng: np.random.Generator,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    excluded: Optional[List[int]] = None,
    shape: int = 1,
) -> npt.NDArray[np.float_]:
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
        size_iter: Union[List, Tuple] = (1,)
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
    figsize: Optional[Tuple[float, float]] = None,
    ax=None,
) -> List[plt.Axes]:
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
    figsize : Optional[Tuple[float, float]], by default None.
        Figure size. Defaults to None.
    ax : matplotlib axes
        Axes on which to plot. Defaults to None.

    Returns
    -------
    List[ax] : matplotlib axes
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


def plot_dependence(*args, kind="pdp", **kwargs):  # pylint: disable=unused-argument
    """
    Partial dependence or individual conditional expectation plot.
    """
    if kind == "pdp":
        warnings.warn(
            "This function has been deprecated. Use plot_pdp instead.",
            FutureWarning,
        )
    elif kind == "ice":
        warnings.warn(
            "This function has been deprecated. Use plot_ice instead.",
            FutureWarning,
        )


def plot_ice(
    bartrv: Variable,
    X: npt.NDArray[np.float_],
    Y: Optional[npt.NDArray[np.float_]] = None,
    var_idx: Optional[List[int]] = None,
    var_discrete: Optional[List[int]] = None,
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
    figsize: Optional[Tuple[float, float]] = None,
    smooth_kwargs: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
) -> List[plt.Axes]:
    """
    Individual conditional expectation plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray[np.float_]
        The covariate matrix.
    Y : Optional[npt.NDArray[np.float_]], by default None.
        The response vector.
    var_idx : Optional[List[int]], by default None.
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : Optional[List[int]], by default None.
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

    fig, axes, shape = _get_axes(bartrv, var_idx, grid, sharey, figsize, ax)

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
        p_d = np.array(y_pred)
        print(p_d.shape)

        for s_i in range(shape):
            if centered:
                p_di = func(p_d[:, :, s_i]) - func(p_d[:, :, s_i][:, 0][:, None])
            else:
                p_di = func(p_d[:, :, s_i])
            if var in var_discrete:
                axes[count].plot(new_x, p_di.mean(0), "o", color=color_mean)
                axes[count].plot(new_x, p_di.T, ".", color=color, alpha=alpha)
            else:
                if smooth:
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
    X: npt.NDArray[np.float_],
    Y: Optional[npt.NDArray[np.float_]] = None,
    xs_interval: str = "quantiles",
    xs_values: Optional[Union[int, List[float]]] = None,
    var_idx: Optional[List[int]] = None,
    var_discrete: Optional[List[int]] = None,
    func: Optional[Callable] = None,
    samples: int = 200,
    random_seed: Optional[int] = None,
    sharey: bool = True,
    smooth: bool = True,
    grid: str = "long",
    color="C0",
    color_mean: str = "C0",
    alpha: float = 0.1,
    figsize: Optional[Tuple[float, float]] = None,
    smooth_kwargs: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
) -> List[plt.Axes]:
    """
    Partial dependence plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray[np.float_]
        The covariate matrix.
    Y : Optional[npt.NDArray[np.float_]], by default None.
        The response vector.
    xs_interval : str
        Method used to compute the values X used to evaluate the predicted function. "linear",
        evenly spaced values in the range of X. "quantiles", the evaluation is done at the specified
        quantiles of X. "insample", the evaluation is done at the values of X.
        For discrete variables these options are ommited.
    xs_values : Optional[Union[int, List[float]]], by default None.
        Values of X used to evaluate the predicted function. If ``xs_interval="linear"`` number of
        points in the evenly spaced grid. If ``xs_interval="quantiles"`` quantile or sequence of
        quantiles to compute, which must be between 0 and 1 inclusive.
        Ignored when ``xs_interval="insample"``.
    var_idx : Optional[List[int]], by default None.
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : Optional[List[int]], by default None.
        List of the indices of the covariate treated as discrete.
    func : Optional[Callable], by default None.
        Arbitrary function to apply to the predictions. Defaults to the identity function.
    samples : int
        Number of posterior samples used in the predictions. Defaults to 200
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

    fig, axes, shape = _get_axes(bartrv, var_idx, grid, sharey, figsize, ax)

    count = 0
    fake_X = _create_pdp_data(X, xs_interval, xs_values)
    for var in range(len(var_idx)):
        excluded = indices[:]
        excluded.remove(var)
        p_d = _sample_posterior(
            all_trees, X=fake_X, rng=rng, size=samples, excluded=excluded, shape=shape
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="hdi currently interprets 2d data")
            new_x = fake_X[:, var]
            for s_i in range(shape):
                p_di = func(p_d[:, :, s_i])
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

    fig.text(-0.05, 0.5, y_label, va="center", rotation="vertical", fontsize=15)

    return axes


def _get_axes(
    bartrv: Variable,
    var_idx: List[int],
    grid: str = "long",
    sharey: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, List[plt.Axes], int]:
    """
    Create and return the figure and axes objects for plotting the variables.

    Partial dependence plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    var_idx : Optional[List[int]], by default None.
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : Optional[List[int]], by default None.
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
    Tuple[plt.Figure, List[plt.Axes], int]
        A tuple containing the figure object, list of axes objects, and the shape value.
    """
    if bartrv.ndim == 1:  # type: ignore
        shape = 1
    else:
        shape = bartrv.eval().shape[0]

    n_plots = len(var_idx) * shape

    if ax is None:
        if grid == "long":
            fig, axes = plt.subplots(n_plots, sharey=sharey, figsize=figsize)
            if n_plots == 1:
                axes = [axes]
        elif grid == "wide":
            fig, axes = plt.subplots(1, n_plots, sharey=sharey, figsize=figsize)
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
    else:
        if isinstance(ax, np.ndarray):
            axes = ax
            fig = ax[0].get_figure()
        else:
            axes = [ax]
            fig = ax.get_figure()  # type: ignore

    return fig, axes, shape


def _prepare_plot_data(
    X: npt.NDArray[np.float_],
    Y: Optional[npt.NDArray[np.float_]] = None,
    xs_interval: str = "quantiles",
    xs_values: Optional[Union[int, List[float]]] = None,
    var_idx: Optional[List[int]] = None,
    var_discrete: Optional[List[int]] = None,
) -> Tuple[
    npt.NDArray[np.float_],
    List[str],
    str,
    List[int],
    List[int],
    List[int],
    str,
    Union[int, None, List[float]],
]:
    """
    Prepare data for plotting.

    Parameters
    ----------
    X : PyTensor Variable, Pandas DataFrame or Numpy array
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

    if hasattr(X, "columns") and hasattr(X, "values"):
        x_names = list(X.columns)
        X = X.values
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
    X: npt.NDArray[np.float_],
    xs_interval: str,
    xs_values: Optional[Union[int, List[float]]] = None,
) -> npt.NDArray[np.float_]:
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
    npt.NDArray[np.float_]
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
    new_x: npt.NDArray[np.float_],
    p_di: npt.NDArray[np.float_],
    kind: str = "pdp",
    smooth_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    smooth_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments for the smoothing function. Defaults to None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
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


def plot_variable_importance(
    idata: az.InferenceData,
    bartrv: Variable,
    X: npt.NDArray[np.float_],
    labels: Optional[List[str]] = None,
    method: str = "VI",
    figsize: Optional[Tuple[float, float]] = None,
    xlabel_angle: float = 0,
    samples: int = 100,
    random_seed: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[List[int], Union[List[plt.Axes], Any]]:
    """
    Estimates variable importance from the BART-posterior.

    Parameters
    ----------
    idata: InferenceData
        InferenceData containing a collection of BART_trees in sample_stats group
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray[np.float_]
        The covariate matrix.
    labels : Optional[List[str]]
        List of the names of the covariates. If X is a DataFrame the names of the covariables will
        be taken from it and this argument will be ignored.
    method : str
        Method used to rank variables. Available options are "VI" (default) and "backward".
        The R squared will be computed following this ranking.
        "VI" counts how many times each variable is included in the posterior distribution
        of trees. "backward" uses a backward search based on the R squared.
        VI requieres less computation time.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    xlabel_angle : float
        rotation angle of the x-axis labels. Defaults to 0. Use values like 45 for
        long labels and/or many variables.
    samples : int
        Number of predictions used to compute correlation for subsets of variables. Defaults to 100
    random_seed : Optional[int]
        random_seed used to sample from the posterior. Defaults to None.
    ax : axes
        Matplotlib axes.

    Returns
    -------
    idxs: indexes of the covariates from higher to lower relative importance
    axes: matplotlib axes
    """
    rng = np.random.default_rng(random_seed)

    all_trees = bartrv.owner.op.all_trees

    if bartrv.ndim == 1:  # type: ignore
        shape = 1
    else:
        shape = bartrv.eval().shape[0]

    if hasattr(X, "columns") and hasattr(X, "values"):
        labels = X.columns
        X = X.values

    n_vars = X.shape[1]

    if figsize is None:
        figsize = (8, 3)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if labels is None:
        labels_ary = np.arange(n_vars).astype(str)
    else:
        labels_ary = np.array(labels)

    ticks = np.arange(n_vars, dtype=int)

    predicted_all = _sample_posterior(
        all_trees, X=X, rng=rng, size=samples, excluded=None, shape=shape
    )

    if method == "VI":
        idxs = np.argsort(
            idata["sample_stats"]["variable_inclusion"].mean(("chain", "draw")).values
        )
        subsets = [idxs[:-i].tolist() for i in range(1, len(idxs))]
        subsets.append(None)  # type: ignore

        indices: List[int] = list(idxs[::-1])

        r2_mean = np.zeros(n_vars)
        r2_hdi = np.zeros((n_vars, 2))
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
                [
                    pearsonr(predicted_all[j].flatten(), predicted_subset[j].flatten())[0] ** 2
                    for j in range(samples)
                ]
            )
            r2_mean[idx] = np.mean(r_2)
            r2_hdi[idx] = az.hdi(r_2)

    elif method == "backward":
        r2_mean = np.zeros(n_vars)
        r2_hdi = np.zeros((n_vars, 2))

        variables = set(range(n_vars))
        least_important_vars: List[int] = []
        indices = []

        # Iterate over each variable to determine its contribution
        # least_important_vars tracks the variable with the lowest contribution
        # at the current stage. One new varible is added at each iteration.
        for i_var in range(n_vars):
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
                    r_2[j] = (
                        pearsonr(predicted_all[j].flatten(), predicted_subset[j].flatten())[0]
                    ) ** 2
                mean_r_2 = np.mean(r_2, dtype=float)
                # Identify the least important combination of variables
                # based on the maximum mean squared Pearson correlation
                if mean_r_2 > max_r_2:
                    max_r_2 = mean_r_2
                    least_important_subset = subset
                    r_2_without_least_important_vars = r_2

            # Save values for plotting later
            r2_mean[i_var] = max_r_2
            r2_hdi[i_var] = az.hdi(r_2_without_least_important_vars)

            # extend current list of least important variable
            least_important_vars += least_important_subset

            # add index of removed variable
            indices += list(set(least_important_subset) - set(indices))

        # add remaining index
        indices += list(set(variables) - set(least_important_vars))

        indices = indices[::-1]
        r2_mean = r2_mean[::-1]
        r2_hdi = r2_hdi[::-1]

    new_labels = [
        "+ " + ele if index != 0 else ele for index, ele in enumerate(labels_ary[indices])
    ]

    r2_yerr_min = np.clip(r2_mean - r2_hdi[:, 0], 0, None)
    r2_yerr_max = np.clip(r2_hdi[:, 1] - r2_mean, 0, None)
    ax.errorbar(
        ticks,
        r2_mean,
        np.array((r2_yerr_min, r2_yerr_max)),
        color="C0",
    )
    ax.axhline(r2_mean[-1], ls="--", color="0.5")
    ax.set_xticks(ticks, new_labels, rotation=xlabel_angle)
    ax.set_ylabel("R²", rotation=0, labelpad=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, n_vars - 0.5)

    return indices, ax


def generate_sequences(n_vars, i_var, include):
    """Generate combinations of variables"""
    if i_var:
        sequences = [tuple(include + [i]) for i in range(n_vars) if i not in include]
    else:
        sequences = [()]
    return sequences
