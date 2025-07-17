import numpy as np
import pymc as pm
import pytest
from numpy.testing import assert_almost_equal
from pymc.initial_point import make_initial_point_fn
from pymc.logprob.basic import transformed_conditional_logp

import pymc_bart as pmb
from pymc_bart.utils import _decode_vi


def assert_moment_is_expected(model, expected, check_finite_logp=True):
    fn = make_initial_point_fn(
        model=model,
        return_transformed=False,
        default_strategy="support_point",
    )
    moment = fn(0)["x"]
    expected = np.asarray(expected)
    try:
        random_draw = model["x"].eval()
    except NotImplementedError:
        random_draw = moment

    assert moment.shape == expected.shape
    assert expected.shape == random_draw.shape
    assert np.allclose(moment, expected)

    if check_finite_logp:
        logp_moment = (
            transformed_conditional_logp(
                (model["x"],),
                rvs_to_values={model["x"]: pm.math.constant(moment)},
                rvs_to_transforms={},
                rvs_to_total_sizes={},
            )[0]
            .sum()
            .eval()
        )
        assert np.isfinite(logp_moment)


@pytest.mark.parametrize(
    argnames="response",
    argvalues=["constant", "linear"],
    ids=["constant", "linear-response"],
)
def test_bart_vi(response):
    X = np.random.normal(0, 1, size=(250, 3))
    Y = np.random.normal(0, 1, size=250)
    X[:, 0] = np.random.normal(Y, 0.1)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=10, response=response)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=Y)
        idata = pm.sample(tune=200, draws=200, random_seed=3415)
        vi_vals = idata["sample_stats"]["variable_inclusion"].values.ravel()
        var_imp = np.array([_decode_vi(val, 3) for val in vi_vals]).sum(axis=0)

        var_imp = var_imp / var_imp.sum()
        assert var_imp[0] > var_imp[1:].sum()
        assert_almost_equal(var_imp.sum(), 1)


@pytest.mark.parametrize(
    argnames="response",
    argvalues=["constant", "linear"],
    ids=["constant", "linear-response"],
)
def test_missing_data(response):
    X = np.random.normal(0, 1, size=(50, 2))
    Y = np.random.normal(0, 1, size=50)
    X[10:20, 0] = np.nan

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=10, response=response)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=Y)
        pm.sample(tune=100, draws=100, chains=1, random_seed=3415)


@pytest.mark.parametrize(
    argnames="response",
    argvalues=["constant", "linear"],
    ids=["constant", "linear-response"],
)
def test_shared_variable(response):
    X = np.random.normal(0, 1, size=(50, 2))
    Y = np.random.normal(0, 1, size=50)

    with pm.Model() as model:
        data_X = pm.Data("data_X", X)
        mu = pmb.BART("mu", data_X, Y, m=2, response=response)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=Y, shape=mu.shape)
        idata = pm.sample(tune=100, draws=100, chains=2, random_seed=3415)
        ppc = pm.sample_posterior_predictive(idata)
        pm.set_data({"data_X": X[:3]})
        ppc2 = pm.sample_posterior_predictive(idata)

    assert ppc.posterior_predictive["y"].shape == (2, 100, 50)
    assert ppc2.posterior_predictive["y"].shape == (2, 100, 3)


@pytest.mark.parametrize(
    argnames="response",
    argvalues=["constant", "linear"],
    ids=["constant", "linear-response"],
)
def test_shape(response):
    X = np.random.normal(0, 1, size=(250, 3))
    Y = np.random.normal(0, 1, size=250)

    with pm.Model() as model:
        w = pmb.BART("w", X, Y, m=2, response=response, shape=(2, 250))
        y = pm.Normal("y", w[0], pm.math.abs(w[1]), observed=Y)
        idata = pm.sample(tune=50, draws=10, random_seed=3415)

    assert model.initial_point()["w"].shape == (2, 250)
    assert idata.posterior.coords["w_dim_0"].data.size == 2
    assert idata.posterior.coords["w_dim_1"].data.size == 250


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


@pytest.mark.parametrize(
    argnames="separate_trees,split_rule",
    argvalues=[
        (False, pmb.ContinuousSplitRule),
        (False, pmb.OneHotSplitRule),
        (False, pmb.SubsetSplitRule),
        (True, pmb.ContinuousSplitRule),
    ],
    ids=["continuous", "one-hot", "subset", "separate-trees"],
)
def test_categorical_model(separate_trees, split_rule):
    Y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    X = np.concatenate([Y[:, None], np.random.randint(0, 6, size=(9, 4))], axis=1)

    with pm.Model() as model:
        lo = pmb.BART(
            "logodds",
            X,
            Y,
            m=2,
            shape=(3, 9),
            split_rules=[split_rule] * 5,
            separate_trees=separate_trees,
        )
        y = pm.Categorical("y", p=pm.math.softmax(lo.T, axis=-1), observed=Y)
        idata = pm.sample(tune=300, draws=300, random_seed=3415)
        idata = pm.sample_posterior_predictive(
            idata, predictions=True, extend_inferencedata=True, random_seed=3415
        )

    # Fit should be good enough so right category is selected over 50% of time
    assert (idata.predictions.y.median(["chain", "draw"]) == Y).all()
    assert pmb.compute_variable_importance(idata, bartrv=lo, X=X)["preds"].shape == (5, 50, 9, 3)


def test_multiple_bart_variables():
    """Test that multiple BART variables can coexist in a single model."""
    X1 = np.random.normal(0, 1, size=(50, 2))
    X2 = np.random.normal(0, 1, size=(50, 3))
    Y = np.random.normal(0, 1, size=50)

    # Create correlated responses
    Y1 = X1[:, 0] + np.random.normal(0, 0.1, size=50)
    Y2 = X2[:, 0] + X2[:, 1] + np.random.normal(0, 0.1, size=50)

    with pm.Model() as model:
        # Two separate BART variables with different covariates
        mu1 = pmb.BART("mu1", X1, Y1, m=5)
        mu2 = pmb.BART("mu2", X2, Y2, m=5)

        # Combined model
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu1 + mu2, sigma, observed=Y)

        # Sample with automatic assignment of BART samplers
        idata = pm.sample(tune=50, draws=50, chains=1, random_seed=3415)

        # Verify both BART variables have their own tree collections
        assert hasattr(mu1.owner.op, "all_trees")
        assert hasattr(mu2.owner.op, "all_trees")

        # Verify trees are stored separately (different object references)
        assert mu1.owner.op.all_trees is not mu2.owner.op.all_trees

        # Verify sampling worked
        assert idata.posterior["mu1"].shape == (1, 50, 50)
        assert idata.posterior["mu2"].shape == (1, 50, 50)

        vi_results = pmb.compute_variable_importance(idata, mu1, X1, model=model)
        assert vi_results["labels"].shape == (2,)
        assert vi_results["preds"].shape == (2, 50, 50)
        assert vi_results["preds_all"].shape == (50, 50)

        vi_tuple = pmb.get_variable_inclusion(idata, X1, model=model, bart_var_name="mu1")
        assert vi_tuple[0].shape == (2,)
        assert len(vi_tuple[1]) == 2
        assert isinstance(vi_tuple[1][0], str)


def test_multiple_bart_variables_manual_step():
    """Test that multiple BART variables work with manually assigned PGBART samplers."""
    X1 = np.random.normal(0, 1, size=(30, 2))
    X2 = np.random.normal(0, 1, size=(30, 2))
    Y = np.random.normal(0, 1, size=30)

    # Create simple responses
    Y1 = X1[:, 0] + np.random.normal(0, 0.1, size=30)
    Y2 = X2[:, 1] + np.random.normal(0, 0.1, size=30)

    with pm.Model() as model:
        # Two separate BART variables
        mu1 = pmb.BART("mu1", X1, Y1, m=3)
        mu2 = pmb.BART("mu2", X2, Y2, m=3)

        # Non-BART variable
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu1 + mu2, sigma, observed=Y)

        # Manually create PGBART samplers for each BART variable
        step1 = pmb.PGBART([mu1], num_particles=5)
        step2 = pmb.PGBART([mu2], num_particles=5)

        # Sample with manual step assignment
        idata = pm.sample(tune=20, draws=20, chains=1, step=[step1, step2], random_seed=3415)

        # Verify both variables were sampled
        assert "mu1" in idata.posterior
        assert "mu2" in idata.posterior
        assert idata.posterior["mu1"].shape == (1, 20, 30)
        assert idata.posterior["mu2"].shape == (1, 20, 30)
