import numpy as np
import pymc as pm
import pytest
from numpy.random import RandomState
from numpy.testing import assert_almost_equal, assert_array_equal
from pymc.tests.distributions.util import assert_moment_is_expected

import pymc_bart as pmb


def test_split_node():
    split_node = pmb.tree.SplitNode(index=5, idx_split_variable=2, split_value=3.0)
    assert split_node.index == 5
    assert split_node.idx_split_variable == 2
    assert split_node.split_value == 3.0
    assert split_node.depth == 2
    assert split_node.get_idx_parent_node() == 2
    assert split_node.get_idx_left_child() == 11
    assert split_node.get_idx_right_child() == 12


def test_leaf_node():
    leaf_node = pmb.tree.LeafNode(index=5, value=3.14, idx_data_points=[1, 2, 3])
    assert leaf_node.index == 5
    assert np.array_equal(leaf_node.idx_data_points, [1, 2, 3])
    assert leaf_node.value == 3.14
    assert leaf_node.get_idx_parent_node() == 2
    assert leaf_node.get_idx_left_child() == 11
    assert leaf_node.get_idx_right_child() == 12


def test_bart_vi():
    X = np.random.normal(0, 1, size=(250, 3))
    Y = np.random.normal(0, 1, size=250)
    X[:, 0] = np.random.normal(Y, 0.1)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=10)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=Y)
        idata = pm.sample(random_seed=3415)
        var_imp = (
            idata.sample_stats["variable_inclusion"]
            .stack(samples=("chain", "draw"))
            .mean("samples")
        )
        var_imp /= var_imp.sum()
        assert var_imp[0] > var_imp[1:].sum()
        assert_almost_equal(var_imp.sum(), 1)


def test_missing_data():
    X = np.random.normal(0, 1, size=(50, 2))
    Y = np.random.normal(0, 1, size=50)
    X[10:20, 0] = np.nan

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=10)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=Y)
        idata = pm.sample(tune=10, draws=10, chains=1, random_seed=3415)


class TestUtils:
    X_norm = np.random.normal(0, 1, size=(50, 2))
    X_binom = np.random.binomial(1, 0.5, size=(50, 1))
    X = np.hstack([X_norm, X_binom])
    Y = np.random.normal(0, 1, size=50)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=10)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=Y)
        idata = pm.sample(random_seed=3415)

    def test_predict(self):
        rng = RandomState(12345)
        pred_all = pmb.predict(self.idata, rng, X=self.X, size=2)
        rng = RandomState(12345)
        pred_first = pmb.predict(self.idata, rng, X=self.X[:10])

        assert_almost_equal(pred_first[0], pred_all[0, :10], decimal=4)
        assert pred_all.shape == (2, 50, 1)
        assert pred_first.shape == (1, 10, 1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {
                "kind": "pdp",
                "samples": 2,
                "xs_interval": "quantiles",
                "xs_values": [0.25, 0.5, 0.75],
                "var_discrete": [3],
            },
            {"kind": "ice", "instances": 2},
            {"var_idx": [0], "rug": False, "smooth": False, "color": "k"},
            {"grid": (1, 2), "sharey": "none", "alpha": 1},
        ],
    )
    def test_pdp(self, kwargs):
        pmb.plot_dependence(self.idata, X=self.X, Y=self.Y, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"labels": ["A", "B", "C"], "samples": 2, "figsize": (6, 6)},
        ],
    )
    def test_vi(self, kwargs):
        pmb.plot_variable_importance(self.idata, X=self.X, **kwargs)

    def test_pdp_pandas_labels(self):
        pd = pytest.importorskip("pandas")

        X_names = ["norm1", "norm2", "binom"]
        X_pd = pd.DataFrame(self.X, columns=X_names)
        Y_pd = pd.Series(self.Y, name="response")
        axes = pmb.plot_dependence(self.idata, X=X_pd, Y=Y_pd)

        figure = axes[0].figure
        assert figure.texts[0].get_text() == "Predicted response"
        assert_array_equal([ax.get_xlabel() for ax in axes], X_names)


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, np.zeros(50)),
    ],
)
def test_bart_moment(size, expected):
    X = np.zeros((50, 2))
    Y = np.zeros(50)
    with pm.Model() as model:
        pmb.BART("x", X=X, Y=Y, size=size)
    assert_moment_is_expected(model, expected)
