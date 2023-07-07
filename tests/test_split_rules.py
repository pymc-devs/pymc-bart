import numpy as np

from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule, SubsetSplitRule
import pytest


@pytest.mark.parametrize(
    argnames="Rule",
    argvalues=[ContinuousSplitRule, OneHotSplitRule, SubsetSplitRule],
    ids=["continuous", "one_hot", "subset"],
)
def test_split_rule(Rule):

    # Should return None if only one available value to pick from
    assert Rule.get_split_value(np.zeros(1)) is None

    # get_split should return a value divide can use
    available_values = np.arange(10).astype(float)
    sv = Rule.get_split_value(available_values)
    left = Rule.divide(available_values, sv)

    # divide should return a boolean numpy array
    # This de facto ensures it is a binary split
    assert len(left) == len(available_values)
    assert left.dtype == "bool"

    # divide should be deterministic
    left_repeated = Rule.divide(available_values, sv)
    assert (left == left_repeated).all()

    # Most elements should have a chance to go either direction
    # NB! This is not 100% necessary, but is a good proxy
    probs = np.array(
        [
            Rule.divide(available_values, Rule.get_split_value(available_values))
            for _ in range(10000)
        ]
    ).mean(axis=0)

    assert (probs > 0.01).sum() >= len(available_values) - 1
    assert (probs < 0.99).sum() >= len(available_values) - 1
