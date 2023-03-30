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
from typing import Dict, Generator, List, Optional

import numpy as np
import numpy.typing as npt
from pytensor import config


class Node:
    """Node of a binary tree.

    Attributes
    ----------
    index : int
    value : float
    idx_data_points : Optional[npt.NDArray[np.int_]]
    idx_split_variable : Optional[npt.NDArray[np.int_]]
    """

    __slots__ = "index", "value", "idx_split_variable", "idx_data_points"

    def __init__(
        self,
        index: int,
        value: float = -1.0,
        idx_data_points: Optional[npt.NDArray[np.int_]] = None,
        idx_split_variable: int = -1,
    ) -> None:
        self.index = index
        self.value = value
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable

    @classmethod
    def new_leaf_node(
        cls, index: int, value: float, idx_data_points: Optional[npt.NDArray[np.int_]]
    ) -> "Node":
        return cls(index, value=value, idx_data_points=idx_data_points)

    @classmethod
    def new_split_node(cls, index: int, split_value: float, idx_split_variable: int) -> "Node":
        return cls(index=index, value=split_value, idx_split_variable=idx_split_variable)

    def get_idx_left_child(self) -> int:
        return self.index * 2 + 1

    def get_idx_right_child(self) -> int:
        return self.index * 2 + 2

    def is_split_node(self) -> bool:
        return self.idx_split_variable >= 0

    def is_leaf_node(self) -> bool:
        return not self.is_split_node()


@lru_cache
def get_depth(index: int) -> int:
    return math.floor(math.log2(index + 1))


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
    idx_leaf_nodes : Optional[npt.NDArray[np.int_]]
        Array with the index of the leaf nodes of the tree.
    output: Optional[npt.NDArray[np.float_]]
        Array of shape number of observations, shape

    Parameters
    ----------
    tree_structure : Dictionary of nodes
    idx_leaf_nodes :  Array with the index of the leaf nodes of the tree.
    output : Array of shape number of observations, shape
    """

    __slots__ = (
        "tree_structure",
        "idx_leaf_nodes",
        "output",
    )

    def __init__(
        self,
        tree_structure: Dict[int, Node],
        idx_leaf_nodes: Optional[npt.NDArray[np.int_]],
        output: npt.NDArray[np.float_],
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
                0: Node.new_leaf_node(
                    index=0, value=leaf_node_value, idx_data_points=idx_data_points
                )
            },
            idx_leaf_nodes=np.array([0]),
            output=np.zeros((num_observations, shape)).astype(config.floatX).squeeze(),
        )

    def __getitem__(self, index) -> Node:
        return self.get_node(index)

    def __setitem__(self, index, node) -> None:
        self.set_node(index, node)

    def copy(self) -> "Tree":
        tree: Dict[int, Node] = {
            k: Node(v.index, v.value, v.idx_data_points, v.idx_split_variable)
            for k, v in self.tree_structure.items()
        }
        idx_leaf_nodes = self.idx_leaf_nodes.copy() if self.idx_leaf_nodes is not None else None
        output = self.output.copy() if self.output is not None else None
        return Tree(tree_structure=tree, idx_leaf_nodes=idx_leaf_nodes, output=output)

    def get_node(self, index: int) -> Node:
        return self.tree_structure[index]

    def set_node(self, index: int, node: Node) -> None:
        self.tree_structure[index] = node
        if node.is_leaf_node() and self.idx_leaf_nodes is not None:
            # self.idx_leaf_nodes.append(index)
            self.idx_leaf_nodes = np.append(self.idx_leaf_nodes, index)

    def grow_leaf_node(
        self, current_node: Node, selected_predictor: int, split_value: float, index_leaf_node: int
    ) -> None:
        current_node.value = split_value
        current_node.idx_split_variable = selected_predictor
        current_node.idx_data_points = None
        if self.idx_leaf_nodes is not None:
            # self.idx_leaf_nodes.remove(index_leaf_node)
            self.idx_leaf_nodes = np.setdiff1d(self.idx_leaf_nodes, index_leaf_node)

    def trim(self) -> "Tree":
        tree: Dict[int, Node] = {
            k: Node(v.index, v.value, None, v.idx_split_variable)
            for k, v in self.tree_structure.items()
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

    def predict(self, x: npt.NDArray[np.float_], excluded: Optional[List[int]] = None) -> float:
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
        float
            Value of the leaf value where the unobserved point lies.
        """
        if excluded is None:
            excluded = []
        return self._traverse_tree(x, 0, excluded)

    def _traverse_tree(
        self, x: npt.NDArray[np.float_], node_index: int, excluded: Optional[List[int]] = None
    ) -> float:
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
        Leaf node value or mean of leaf node values
        """
        current_node: Node = self.get_node(node_index)
        if current_node.is_leaf_node():
            return current_node.value

        if excluded is not None and current_node.idx_split_variable in excluded:
            leaf_values: List[float] = []
            self._traverse_leaf_values(leaf_values, node_index)
            return np.mean(leaf_values, axis=0)

        if x[current_node.idx_split_variable] <= current_node.value:
            next_node = current_node.get_idx_left_child()
        else:
            next_node = current_node.get_idx_right_child()
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
            self._traverse_leaf_values(leaf_values, node.get_idx_left_child())
            self._traverse_leaf_values(leaf_values, node.get_idx_right_child())
