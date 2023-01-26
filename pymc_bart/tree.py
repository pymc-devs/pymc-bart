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

import math

from functools import lru_cache

from pytensor import config
import numpy as np


class Tree:
    """Full binary tree.

    A full binary tree is a tree where each node has exactly zero or two children.
    This structure is used as the basic component of the Bayesian Additive Regression Tree (BART)

    Attributes
    ----------
    tree_structure : dict
        A dictionary that represents the nodes stored in breadth-first order, based in the array
        method for storing binary trees (https://en.wikipedia.org/wiki/Binary_tree#Arrays).
        The dictionary's keys are integers that represent the nodes position.
        The dictionary's values are objects of type Node that represent the split and leaf nodes
        of the tree itself.
    idx_leaf_nodes : list
        List with the index of the leaf nodes of the tree.
    output: array
        Array of shape number of observations, shape

    Parameters
    ----------
    tree_structure : Dictionary of nodes
    idx_leaf_nodes :  List with the index of the leaf nodes of the tree.
    output : Array of shape number of observations, shape
    """

    __slots__ = (
        "tree_structure",
        "idx_leaf_nodes",
        "output",
    )

    def __init__(self, tree_structure, idx_leaf_nodes, output):
        self.tree_structure = tree_structure
        self.idx_leaf_nodes = idx_leaf_nodes
        self.output = output

    @classmethod
    def new_tree(cls, leaf_node_value, idx_data_points, num_observations, shape):
        return cls(
            tree_structure={
                0: Node.new_leaf_node(0, value=leaf_node_value, idx_data_points=idx_data_points)
            },
            idx_leaf_nodes=[0],
            output=np.zeros((shape, num_observations)).astype(config.floatX).squeeze(),
        )

    def __getitem__(self, index):
        return self.get_node(index)

    def __setitem__(self, index, node):
        self.set_node(index, node)

    def copy(self):
        tree = {
            k: Node(v.index, v.value, v.idx_data_points, v.idx_split_variable)
            for k, v in self.tree_structure.items()
        }
        return Tree(tree, self.idx_leaf_nodes.copy(), self.output.copy())

    def get_node(self, index) -> "Node":
        return self.tree_structure[index]

    def set_node(self, index, node):
        self.tree_structure[index] = node
        if node.is_leaf_node():
            self.idx_leaf_nodes.append(index)

    def delete_leaf_node(self, index):
        self.idx_leaf_nodes.remove(index)
        del self.tree_structure[index]

    def trim(self):
        tree = {
            k: Node(v.index, v.value, None, v.idx_split_variable)
            for k, v in self.tree_structure.items()
        }
        return Tree(tree, None, None)

    def get_split_variables(self):
        return [
            node.idx_split_variable for node in self.tree_structure.values() if node.is_split_node()
        ]

    def _predict(self):
        output = self.output
        for node_index in self.idx_leaf_nodes:
            leaf_node = self.get_node(node_index)
            output[leaf_node.idx_data_points] = leaf_node.value
        return output

    def predict(self, x, excluded=None):
        """
        Predict output of tree for an (un)observed point x.

        Parameters
        ----------
        x : numpy array
            Unobserved point
        excluded: list
                Indexes of the variables to exclude when computing predictions

        Returns
        -------
        float
            Value of the leaf value where the unobserved point lies.
        """
        if excluded is None:
            excluded = []
        return self._traverse_tree(x, 0, excluded)

    def _traverse_tree(self, x, node_index, excluded):
        """
        Traverse the tree starting from a particular node given an unobserved point.

        Parameters
        ----------
        x : np.ndarray
        node_index : int

        Returns
        -------
        Leaf node value or mean of leaf node values
        """
        current_node = self.get_node(node_index)
        if current_node.is_leaf_node():
            return current_node.value
        if current_node.idx_split_variable in excluded:
            leaf_values = []
            self._traverse_leaf_values(leaf_values, node_index)
            return np.mean(leaf_values, 0)

        if x[current_node.idx_split_variable] <= current_node.value:
            left_child = current_node.get_idx_left_child()
            return self._traverse_tree(x, left_child, excluded)
        else:
            right_child = current_node.get_idx_right_child()
            return self._traverse_tree(x, right_child, excluded)

    def _traverse_leaf_values(self, leaf_values, node_index):
        """
        Traverse the tree appending leaf values starting from a particular node.

        Parameters
        ----------
        node_index : int

        Returns
        -------
        List of leaf node values
        """
        node = self.get_node(node_index)
        if node.is_leaf_node():
            leaf_values.append(node.value)
        else:
            self._traverse_leaf_values(leaf_values, node.get_idx_left_child())
            self._traverse_leaf_values(leaf_values, node.get_idx_right_child())


class Node:
    __slots__ = "index", "value", "idx_split_variable", "idx_data_points"

    def __init__(self, index: int, value=-1, idx_data_points=None, idx_split_variable=-1):
        self.index = index
        self.value = value
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable

    @classmethod
    def new_leaf_node(cls, index: int, value, idx_data_points) -> "Node":
        return cls(index, value=value, idx_data_points=idx_data_points)

    @classmethod
    def new_split_node(cls, index: int, split_value, idx_split_variable) -> "Node":
        return cls(index, value=split_value, idx_split_variable=idx_split_variable)

    def get_idx_parent_node(self) -> int:
        return (self.index - 1) // 2

    def get_idx_left_child(self) -> int:
        return self.index * 2 + 1

    def get_idx_right_child(self) -> int:
        return self.get_idx_left_child() + 1

    def is_split_node(self) -> bool:
        return self.idx_split_variable >= 0

    def is_leaf_node(self) -> bool:
        return not self.is_split_node()


@lru_cache
def get_depth(index: int) -> int:
    return math.floor(math.log2(index + 1))
