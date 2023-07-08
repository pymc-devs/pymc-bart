import numpy as np

from pymc_bart.tree import Node, get_idx_left_child, get_idx_right_child, get_depth


def test_split_node():
    index = 5
    split_node = Node(idx_split_variable=2, value=3.0)
    assert get_depth(index) == 2
    assert split_node.value == 3.0
    assert split_node.idx_split_variable == 2
    assert split_node.idx_data_points is None
    assert get_idx_left_child(index) == 11
    assert get_idx_right_child(index) == 12
    assert split_node.is_split_node() is True
    assert split_node.is_leaf_node() is False


def test_leaf_node():
    index = 5
    leaf_node = Node.new_leaf_node(value=3.14, idx_data_points=[1, 2, 3])
    assert get_depth(index) == 2
    assert leaf_node.value == 3.14
    assert leaf_node.idx_split_variable == -1
    assert np.array_equal(leaf_node.idx_data_points, [1, 2, 3])
    assert get_idx_left_child(index) == 11
    assert get_idx_right_child(index) == 12
    assert leaf_node.is_split_node() is False
    assert leaf_node.is_leaf_node() is True
