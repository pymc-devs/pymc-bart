import numpy as np

from pymc_bart.tree import Node, get_depth


def test_split_node():
    split_node = Node.new_split_node(index=5, idx_split_variable=2, split_value=3.0)
    assert split_node.index == 5
    assert get_depth(split_node.index) == 2
    assert split_node.value == 3.0
    assert split_node.idx_split_variable == 2
    assert split_node.idx_data_points is None
    assert split_node.get_idx_parent_node() == 2
    assert split_node.get_idx_left_child() == 11
    assert split_node.get_idx_right_child() == 12
    assert split_node.is_split_node() is True
    assert split_node.is_leaf_node() is False


def test_leaf_node():
    leaf_node = Node.new_leaf_node(index=5, value=3.14, idx_data_points=[1, 2, 3])
    assert leaf_node.index == 5
    assert get_depth(leaf_node.index) == 2
    assert leaf_node.value == 3.14
    assert leaf_node.idx_split_variable == -1
    assert np.array_equal(leaf_node.idx_data_points, [1, 2, 3])
    assert leaf_node.get_idx_parent_node() == 2
    assert leaf_node.get_idx_left_child() == 11
    assert leaf_node.get_idx_right_child() == 12
    assert leaf_node.is_split_node() is False
    assert leaf_node.is_leaf_node() is True
