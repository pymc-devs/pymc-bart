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

from functools import lru_cache
from typing import Dict, Generator, List, Optional

import numpy as np
import numpy.typing as npt
from pytensor import config


class Node:
    """Node of a binary tree.

    Attributes
    ----------
    value : float
    idx_data_points : Optional[npt.NDArray[np.int_]]
    idx_split_variable : Optional[npt.NDArray[np.int_]]
    """

    __slots__ = "value", "idx_split_variable", "idx_data_points"

    def __init__(
        self,
        value: float = -1.0,
        idx_data_points: Optional[npt.NDArray[np.int_]] = None,
        idx_split_variable: int = -1,
    ) -> None:
        self.value = value
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable

    @classmethod
    def new_leaf_node(cls, value: float, idx_data_points: Optional[npt.NDArray[np.int_]]) -> "Node":
        return cls(value=value, idx_data_points=idx_data_points)

    @classmethod
    def new_split_node(cls, split_value: float, idx_split_variable: int) -> "Node":
        return cls(value=split_value, idx_split_variable=idx_split_variable)

    def is_split_node(self) -> bool:
        return self.idx_split_variable >= 0

    def is_leaf_node(self) -> bool:
        return not self.is_split_node()


def get_idx_left_child(index) -> int:
    return index * 2 + 1


def get_idx_right_child(index) -> int:
    return index * 2 + 2


@lru_cache
def get_depth(index: int) -> int:
    return (index + 1).bit_length() - 1


class Tree:
    """Full binary tree.

    A full binary tree is a tree where each node has exactly zero or two children.
    This structure is used as the basic component of the Bayesian Additive Regression Tree (BART)

    Attributes
    ----------
    tree_structure : Dict[int, Node]
        A dictionary that represents the nodes stored in breadth-first order, based in the array
        method for storing binary trees (https://en.wikipedia.org/wiki/Binary_tree#Arrays).
        The dictionary's keys are integers that represent the nodes position.
        The dictionary's values are objects of type Node that represent the split and leaf nodes
        of the tree itself.
    output: Optional[npt.NDArray[np.float_]]
        Array of shape number of observations, shape
    idx_leaf_nodes : Optional[List[int]], by default None.
        Array with the index of the leaf nodes of the tree.

    Parameters
    ----------
    tree_structure : Dictionary of nodes
    output : Array of shape number of observations, shape
    idx_leaf_nodes :  List with the index of the leaf nodes of the tree.
    """

    __slots__ = (
        "tree_structure",
        "output",
        "idx_leaf_nodes",
    )

    def __init__(
        self,
        tree_structure: Dict[int, Node],
        output: npt.NDArray[np.float_],
        idx_leaf_nodes: Optional[List[int]] = None,
    ) -> None:
        self.tree_structure = tree_structure
        self.idx_leaf_nodes = idx_leaf_nodes
        self.output = output

    @classmethod
    def new_tree(
        cls,
        leaf_node_value: float,
        idx_data_points: Optional[npt.NDArray[np.int_]],
        num_observations: int,
        shape: int,
    ) -> "Tree":
        return cls(
            tree_structure={
                0: Node.new_leaf_node(value=leaf_node_value, idx_data_points=idx_data_points)
            },
            idx_leaf_nodes=[0],
            output=np.zeros((num_observations, shape)).astype(config.floatX).squeeze(),
        )

    def __getitem__(self, index) -> Node:
        return self.get_node(index)

    def __setitem__(self, index, node) -> None:
        self.set_node(index, node)

    def copy(self) -> "Tree":
        tree: Dict[int, Node] = {
            k: Node(v.value, v.idx_data_points, v.idx_split_variable)
            for k, v in self.tree_structure.items()
        }
        idx_leaf_nodes = self.idx_leaf_nodes.copy() if self.idx_leaf_nodes is not None else None
        return Tree(tree_structure=tree, idx_leaf_nodes=idx_leaf_nodes, output=self.output)

    def get_node(self, index: int) -> Node:
        return self.tree_structure[index]

    def set_node(self, index: int, node: Node) -> None:
        self.tree_structure[index] = node
        if node.is_leaf_node() and self.idx_leaf_nodes is not None:
            self.idx_leaf_nodes.append(index)

    def grow_leaf_node(
        self, current_node: Node, selected_predictor: int, split_value: float, index_leaf_node: int
    ) -> None:
        current_node.value = split_value
        current_node.idx_split_variable = selected_predictor
        current_node.idx_data_points = None
        if self.idx_leaf_nodes is not None:
            self.idx_leaf_nodes.remove(index_leaf_node)

    def trim(self) -> "Tree":
        tree: Dict[int, Node] = {
            k: Node(v.value, None, v.idx_split_variable) for k, v in self.tree_structure.items()
        }
        return Tree(tree_structure=tree, idx_leaf_nodes=None, output=np.array([-1]))

    def get_split_variables(self) -> Generator[int, None, None]:
        for node in self.tree_structure.values():
            if node.is_split_node():
                yield node.idx_split_variable

    def _predict(self) -> npt.NDArray[np.float_]:
        output = self.output

        if self.idx_leaf_nodes is not None:
            for node_index in self.idx_leaf_nodes:
                leaf_node = self.get_node(node_index)
                output[leaf_node.idx_data_points] = leaf_node.value
        return output.T

    def predict(
        self, x: npt.NDArray[np.float_], excluded: Optional[List[int]] = None
    ) -> npt.NDArray[np.float_]:
        """
        Predict output of tree for an (un)observed point x.

        Parameters
        ----------
        x : npt.NDArray[np.float_]
            Unobserved point
        excluded: Optional[List[int]]
            Indexes of the variables to exclude when computing predictions

        Returns
        -------
        npt.NDArray[np.float_]
            Value of the leaf value where the unobserved point lies.
        """
        if excluded is None:
            excluded = []
        return self._traverse_tree(x, 0, excluded)

    def _traverse_tree(
        self,
        x: npt.NDArray[np.float_],
        node_index: int,
        excluded: Optional[List[int]] = None,
    ) -> npt.NDArray[np.float_]:
        """
        Traverse the tree starting from a particular node given an unobserved point.

        Parameters
        ----------
        x : npt.NDArray[np.float_]
            Unobserved point
        node_index : int
            Index of the node to start the traversal from
        excluded: Optional[List[int]]
            Indexes of the variables to exclude when computing predictions

        Returns
        -------
        npt.NDArray[np.float_]
            Leaf node value or mean of leaf node values
        """
        current_node: Node = self.get_node(node_index)
        if current_node.is_leaf_node():
            return np.array(current_node.value)

        if excluded is not None and current_node.idx_split_variable in excluded:
            leaf_values: List[float] = []
            self._traverse_leaf_values(leaf_values, node_index)
            return np.mean(leaf_values, axis=0)

        if x[current_node.idx_split_variable] <= current_node.value:
            next_node = get_idx_left_child(node_index)
        else:
            next_node = get_idx_right_child(node_index)
        return self._traverse_tree(x=x, node_index=next_node, excluded=excluded)

    def _traverse_leaf_values(self, leaf_values: List[float], node_index: int) -> None:
        """
        Traverse the tree appending leaf values starting from a particular node.

        Parameters
        ----------
        leaf_values : List[float]
        node_index : int
        """
        node = self.get_node(node_index)
        if node.is_leaf_node():
            leaf_values.append(node.value)
        else:
            self._traverse_leaf_values(leaf_values, get_idx_left_child(node_index))
            self._traverse_leaf_values(leaf_values, get_idx_right_child(node_index))
