from unittest import TestCase
import pytest
import numpy as np
import pymc as pm

import pymc_bart as pmb
from pymc_bart.pgbart import (
    NormalSampler,
    UniformSampler,
    discrete_uniform_sampler,
    fast_mean,
    fast_linear_fit,
)


class TestSystematic(TestCase):
    def test_systematic(self):
        X = np.random.normal(0, 1, size=(250, 3))
        Y = np.random.normal(0, 1, size=250)
        X[:, 0] = np.random.normal(Y, 0.1)

        with pm.Model() as model:
            mu = pmb.BART("mu", X, Y, m=10)
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", mu, sigma, observed=Y)
            step = pmb.PGBART([mu])

        normalized_weights = np.array([0.5, 0.3, 0.2])
        indices = step.systematic(normalized_weights)

        self.assertEqual(len(indices), len(normalized_weights))
        self.assertEqual(indices.dtype, np.int_)
        self.assertTrue(all(i >= 0 and i < len(normalized_weights) for i in indices))

        normalized_weights = np.array([0, 0.25, 0.75])
        indices = step.systematic(normalized_weights)
        self.assertTrue(all(i >= 1 and i < len(normalized_weights) for i in indices))


def test_fast_mean():
    values = np.random.uniform(size=10)
    np.testing.assert_almost_equal(fast_mean(values), np.mean(values))

    values = np.random.uniform(size=(2, 10))
    np.testing.assert_array_almost_equal(fast_mean(values), np.mean(values, 1))


@pytest.mark.parametrize(
    argnames="x,y,a_expected, b_expected",
    argvalues=[
        (np.array([1, 2, 3, 4, 5]), np.array([[1, 2, 3, 4, 5]]), 0.0, 1.0),
        (np.array([1, 2, 3, 4, 5]), np.array([[1, 1, 1, 1, 1]]), 1.0, 0.0),
    ],
    ids=["1d-id", "1d-const"],
)
def test_fast_linear_fit(x, y, a_expected, b_expected):
    y_fit, linear_params = fast_linear_fit(x, y, m=1, norm=np.zeros(1))
    assert linear_params[0] == a_expected
    assert linear_params[1] == b_expected
    np.testing.assert_almost_equal(
        actual=y_fit, desired=np.atleast_2d(a_expected + x * b_expected).T
    )


def test_discrete_uniform():
    sample = discrete_uniform_sampler(7)
    assert isinstance(sample, int)
    samples = np.array([discrete_uniform_sampler(7) for i in range(1000)])
    assert all(samples >= 0)
    assert all(samples < 7)


def test_normal_sampler():
    normal = NormalSampler(2, shape=1)
    samples = np.array([normal.rvs() for i in range(100000)])
    np.testing.assert_almost_equal(samples.mean(), 0, decimal=2)
    np.testing.assert_almost_equal(samples.std(), 2, decimal=2)

    normal = NormalSampler(2, shape=2)
    samples = np.array([normal.rvs() for i in range(100000)])
    np.testing.assert_almost_equal(samples.mean(0), [0, 0], decimal=2)
    np.testing.assert_almost_equal(samples.std(0), [2, 2], decimal=2)


def test_uniform_sampler():
    uniform = UniformSampler(0.5, 2, shape=1)
    samples = np.array([uniform.rvs() for i in range(100000)])
    np.testing.assert_almost_equal(samples.mean(), 1.25, decimal=2)
    np.testing.assert_almost_equal(samples.std(), 0.43, decimal=2)

    uniform = UniformSampler(0.5, 2, shape=2)
    samples = np.array([uniform.rvs() for i in range(100000)])
    np.testing.assert_almost_equal(samples.mean(0), [1.25, 1.25], decimal=2)
    np.testing.assert_almost_equal(samples.std(0), [0.43, 0.43], decimal=2)
