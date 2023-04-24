import numpy as np

from pymc_bart.tree import  get_idx_left_child, get_idx_right_child, get_depth, is_leaf_node, new_leaf_node


def test_leaf_node():
    index = 5
    leaf_node = new_leaf_node(value=3.14, idx_data_points=[1, 2, 3])
    assert get_depth(index) == 2
    assert leaf_node.value == 3.14
    assert leaf_node.idx_split_variable == -1
    assert np.array_equal(leaf_node.idx_data_points, [1, 2, 3])
    assert get_idx_left_child(index) == 11
    assert get_idx_right_child(index) == 12
    assert is_leaf_node(leaf_node) is True
