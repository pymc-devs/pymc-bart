# pylint: disable=unused-argument
# pylint: disable=arguments-differ
#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import warnings
from multiprocessing import Manager
from typing import Optional

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pandas import DataFrame, Series
from pymc.distributions.distribution import Distribution, _support_point
from pymc.logprob.abstract import _logprob
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.variable import TensorVariable

from .split_rules import SplitRule
from .tree import Tree
from .utils import TensorLike, _sample_posterior

__all__ = ["BART"]


class BARTRV(RandomVariable):
    """Base class for BART."""

    name: str = "BART"
    signature = "(m,n),(m),(),(),() -> (m)"
    dtype: str = "floatX"
    _print_name: tuple[str, str] = ("BART", "\\operatorname{BART}")
    all_trees = list[list[list[Tree]]]

    def _supp_shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):  # pylint: disable=arguments-renamed
        idx = dist_params[0].ndim - 2
        return [dist_params[0].shape[idx]]

    @classmethod
    def rng_fn(  # pylint: disable=W0237
        cls, rng=None, X=None, Y=None, m=None, alpha=None, beta=None, size=None
    ):
        if not size:
            size = None

        if not cls.all_trees:
            if isinstance(cls.Y, (TensorSharedVariable, TensorVariable)):
                Y = cls.Y.eval()
            else:
                Y = cls.Y

            if size is not None:
                return np.full((size[0], Y.shape[0]), Y.mean())
            else:
                return np.full(Y.shape[0], Y.mean())
        else:
            if size is not None:
                shape = size[0]
            else:
                shape = 1
            return _sample_posterior(cls.all_trees, cls.X, rng=rng, shape=shape).squeeze().T


bart = BARTRV()


class BART(Distribution):
    r"""
    Bayesian Additive Regression Tree distribution.

    Distribution representing a sum over trees

    Parameters
    ----------
    X : PyTensor Variable, Pandas/Polars DataFrame or Numpy array
        The covariate matrix.
    Y : PyTensor Variable, Pandas/Polar DataFrame/Series,or Numpy array
        The response vector.
    m : int
        Number of trees.
    response : str
        How the leaf_node values are computed. Available options are ``constant``, ``linear`` or
        ``mix``. Defaults to ``constant``. Options ``linear`` and ``mix`` are still experimental.
    alpha : float
        Controls the prior probability over the depth of the trees.
        Should be in the (0, 1) interval.
    beta : float
        Controls the prior probability over the number of leaves of the trees.
        Should be positive.
    split_prior : Optional[list[float]], default None.
        List of positive numbers, one per column in input data.
        Defaults to None, all covariates have the same prior probability to be selected.
    split_rules : Optional[list[SplitRule]], default None
        List of SplitRule objects, one per column in input data.
        Allows using different split rules for different columns. Default is ContinuousSplitRule.
        Other options are OneHotSplitRule and SubsetSplitRule, both meant for categorical variables.
    separate_trees : Optional[bool], default False
        When training multiple trees (by setting a shape parameter), the default behavior is to
        learn a joint tree structure and only have different leaf values for each.
        This flag forces a fully separate tree structure to be trained instead.
        This is unnecessary in many cases and is considerably slower, multiplying
        run-time roughly by number of dimensions.

    Notes
    -----
    The parameters ``alpha`` and ``beta`` parametrize the probability that a node at
    depth :math:`d \: (= 0, 1, 2,...)` is non-terminal, given by :math:`\alpha(1 + d)^{-\beta}`.
    The default values are :math:`\alpha = 0.95` and :math:`\beta = 2`.

    This is the recommend prior by Chipman Et al. BART: Bayesian additive regression trees,
    `link <https://doi.org/10.1214/09-AOAS285>`__
    """

    def __new__(
        cls,
        name: str,
        X: TensorLike,
        Y: TensorLike,
        m: int = 50,
        alpha: float = 0.95,
        beta: float = 2.0,
        response: str = "constant",
        split_prior: Optional[npt.NDArray] = None,
        split_rules: Optional[list[SplitRule]] = None,
        separate_trees: Optional[bool] = False,
        **kwargs,
    ):
        if response in ["linear", "mix"]:
            warnings.warn(
                "Options linear and mix are experimental and still not well tested\n"
                + "Use with caution."
            )
        manager = Manager()
        cls.all_trees = manager.list()

        X, Y = preprocess_xy(X, Y)

        split_prior = np.array([]) if split_prior is None else np.asarray(split_prior)

        bart_op = type(
            f"BART_{name}",
            (BARTRV,),
            {
                "name": "BART",
                "all_trees": cls.all_trees,
                "inplace": False,
                "initval": Y.mean(),
                "X": X,
                "Y": Y,
                "m": m,
                "response": response,
                "alpha": alpha,
                "beta": beta,
                "split_prior": split_prior,
                "split_rules": split_rules,
                "separate_trees": separate_trees,
            },
        )()

        Distribution.register(BARTRV)

        @_support_point.register(BARTRV)
        def get_moment(rv, size, *rv_inputs):
            return cls.get_moment(rv, size, *rv_inputs)

        cls.rv_op = bart_op
        params = [X, Y, m, alpha, beta]
        return super().__new__(cls, name, *params, **kwargs)

    @classmethod
    def dist(cls, *params, **kwargs):
        return super().dist(params, **kwargs)

    def logp(self, x, *inputs):
        """Calculate log probability.

        Parameters
        ----------
        x: numeric, TensorVariable
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        return pt.zeros_like(x)

    @classmethod
    def get_moment(cls, rv, size, *rv_inputs):
        mean = pt.fill(size, rv.Y.mean())
        return mean


def preprocess_xy(X: TensorLike, Y: TensorLike) -> tuple[npt.NDArray, npt.NDArray]:
    if isinstance(Y, (Series, DataFrame)):
        Y = Y.to_numpy()
    if isinstance(X, (Series, DataFrame)):
        X = X.to_numpy()

    try:
        import polars as pl

        if isinstance(X, (pl.Series, pl.DataFrame)):
            X = X.to_numpy()
        if isinstance(Y, (pl.Series, pl.DataFrame)):
            Y = Y.to_numpy()
    except ImportError:
        pass

    Y = Y.astype(float)
    X = X.astype(float)

    return X, Y


@_logprob.register(BARTRV)
def logp(op, value_var, *dist_params, **kwargs):
    _dist_params = dist_params[3:]
    value_var = value_var[0]
    return BART.logp(value_var, *_dist_params)  # pylint: disable=no-value-for-parameter
