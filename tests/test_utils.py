import numpy as np
import pymc as pm
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

import pymc_bart as pmb


class TestUtils:
    X_norm = np.random.normal(0, 1, size=(50, 2))
    X_binom = np.random.binomial(1, 0.5, size=(50, 1))
    X = np.hstack([X_norm, X_binom])
    Y = np.random.normal(0, 1, size=50)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=10)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=Y)
        idata = pm.sample(tune=200, draws=200, random_seed=3415)

    def test_sample_posterior(self):
        all_trees = self.mu.owner.op.all_trees
        rng = np.random.default_rng(3)
        pred_all = pmb.utils._sample_posterior(all_trees, X=self.X, rng=rng, size=2)
        rng = np.random.default_rng(3)
        pred_first = pmb.utils._sample_posterior(all_trees, X=self.X[:10], rng=rng)

        assert_almost_equal(pred_first[0], pred_all[0, :10], decimal=4)
        assert pred_all.shape == (2, 50, 1)
        assert pred_first.shape == (1, 10, 1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {
                "samples": 2,
                "var_discrete": [3],
            },
            {"instances": 2},
            {"var_idx": [0], "smooth": False, "color": "k"},
            {"grid": (1, 2), "sharey": "none", "alpha": 1},
            {"var_discrete": [0]},
        ],
    )
    def test_ice(self, kwargs):
        pmb.plot_ice(self.mu, X=self.X, Y=self.Y, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {
                "samples": 2,
                "xs_interval": "quantiles",
                "xs_values": [0.25, 0.5, 0.75],
                "var_discrete": [3],
            },
            {"var_idx": [0], "smooth": False, "color": "k"},
            {"grid": (1, 2), "sharey": "none", "alpha": 1},
            {"var_discrete": [0]},
        ],
    )
    def test_pdp(self, kwargs):
        pmb.plot_pdp(self.mu, X=self.X, Y=self.Y, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"samples": 50},
            {"labels": ["A", "B", "C"], "samples": 2, "figsize": (6, 6)},
        ],
    )
    def test_vi(self, kwargs):
        samples = kwargs.pop("samples")
        vi_results = pmb.compute_variable_importance(
            self.idata, bartrv=self.mu, X=self.X, samples=samples
        )
        pmb.plot_variable_importance(vi_results, **kwargs)
        pmb.plot_scatter_submodels(vi_results, **kwargs)

        user_terms = pmb.vi_to_kulprit(vi_results)
        assert len(user_terms) == 3
        assert all("+" not in term for terms in user_terms[1:] for term in terms)

    def test_pdp_pandas_labels(self):
        pd = pytest.importorskip("pandas")

        X_names = ["norm1", "norm2", "binom"]
        X_pd = pd.DataFrame(self.X, columns=X_names)
        Y_pd = pd.Series(self.Y, name="response")
        axes = pmb.plot_pdp(self.mu, X=X_pd, Y=Y_pd)

        figure = axes[0].figure
        assert figure.texts[0].get_text() == "Partial response"
        assert_array_equal([ax.get_xlabel() for ax in axes], X_names)


def test_encoder_decoder():
    """Test that the encoder-decoder works correctly."""
    test_cases = [
        np.zeros(3, dtype=int),
        np.ones(10, dtype=int),
        np.array([4, 0, 1, 0, 2, 0, 3, 0, 0, 0]),
        np.array([100, 50, 0, 1]),
        np.array([1, 2, 4, 8, 16]),
    ]
    for case in test_cases:
        encoded = pmb.utils._encode_vi(case)
        decoded = pmb.utils._decode_vi(encoded, len(case))
        assert np.array_equal(decoded, case)
