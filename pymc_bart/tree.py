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
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from pytensor import config

from .split_rules import SplitRule


class Node:
    """Node of a binary tree.

    Attributes
    ----------
    value : npt.NDArray[np.float_]
    idx_data_points : Optional[npt.NDArray[np.int_]]
    idx_split_variable : int
    linear_params: Optional[List[float]] = None
    """

    __slots__ = "value", "nvalue", "idx_split_variable", "idx_data_points", "linear_params"

    def __init__(
        self,
        value: npt.NDArray[np.float_] = np.array([-1.0]),
        nvalue: int = 0,
        idx_data_points: Optional[npt.NDArray[np.int_]] = None,
        idx_split_variable: int = -1,
        linear_params: Optional[List[npt.NDArray[np.float_]]] = None,
    ) -> None:
        self.value = value
        self.nvalue = nvalue
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable
        self.linear_params = linear_params

    @classmethod
    def new_leaf_node(
        cls,
        value: npt.NDArray[np.float_],
        nvalue: int = 0,
        idx_data_points: Optional[npt.NDArray[np.int_]] = None,
        idx_split_variable: int = -1,
        linear_params: Optional[List[npt.NDArray[np.float_]]] = None,
    ) -> "Node":
        return cls(
            value=value,
            nvalue=nvalue,
            idx_data_points=idx_data_points,
            idx_split_variable=idx_split_variable,
            linear_params=linear_params,
        )

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
    split_rules : List[SplitRule]
        List of SplitRule objects, one per column in input data.
        Allows using different split rules for different columns. Default is ContinuousSplitRule.
        Other options are OneHotSplitRule and SubsetSplitRule, both meant for categorical variables.
    idx_leaf_nodes : Optional[List[int]], by default None.
        Array with the index of the leaf nodes of the tree.

    Parameters
    ----------
    tree_structure : Dictionary of nodes
    output : Array of shape number of observations, shape
    idx_leaf_nodes :  List with the index of the leaf nodes of the tree.
    """

    __slots__ = ("tree_structure", "output", "idx_leaf_nodes", "split_rules")

    def __init__(
        self,
        tree_structure: Dict[int, Node],
        output: npt.NDArray[np.float_],
        split_rules: List[SplitRule],
        idx_leaf_nodes: Optional[List[int]] = None,
    ) -> None:
        self.tree_structure = tree_structure
        self.idx_leaf_nodes = idx_leaf_nodes
        self.split_rules = split_rules
        self.output = output

    @classmethod
    def new_tree(
        cls,
        leaf_node_value: npt.NDArray[np.float_],
        idx_data_points: Optional[npt.NDArray[np.int_]],
        num_observations: int,
        shape: int,
        split_rules: List[SplitRule],
    ) -> "Tree":
        return cls(
            tree_structure={
                0: Node.new_leaf_node(
                    value=leaf_node_value,
                    nvalue=len(idx_data_points) if idx_data_points is not None else 0,
                    idx_data_points=idx_data_points,
                )
            },
            idx_leaf_nodes=[0],
            output=np.zeros((num_observations, shape)).astype(config.floatX),
            split_rules=split_rules,
        )

    def __getitem__(self, index) -> Node:
        return self.get_node(index)

    def __setitem__(self, index, node) -> None:
        self.set_node(index, node)

    def copy(self) -> "Tree":
        tree: Dict[int, Node] = {
            k: Node(
                value=v.value,
                nvalue=v.nvalue,
                idx_data_points=v.idx_data_points,
                idx_split_variable=v.idx_split_variable,
                linear_params=v.linear_params,
            )
            for k, v in self.tree_structure.items()
        }
        idx_leaf_nodes = self.idx_leaf_nodes.copy() if self.idx_leaf_nodes is not None else None
        return Tree(
            tree_structure=tree,
            idx_leaf_nodes=idx_leaf_nodes,
            output=self.output,
            split_rules=self.split_rules,
        )

    def get_node(self, index: int) -> Node:
        return self.tree_structure[index]

    def set_node(self, index: int, node: Node) -> None:
        self.tree_structure[index] = node
        if node.is_leaf_node() and self.idx_leaf_nodes is not None:
            self.idx_leaf_nodes.append(index)

    def grow_leaf_node(
        self,
        current_node: Node,
        selected_predictor: int,
        split_value: npt.NDArray[np.float_],
        index_leaf_node: int,
    ) -> None:
        current_node.value = split_value
        current_node.idx_split_variable = selected_predictor
        current_node.idx_data_points = None
        if self.idx_leaf_nodes is not None:
            self.idx_leaf_nodes.remove(index_leaf_node)

    def trim(self) -> "Tree":
        tree: Dict[int, Node] = {
            k: Node(
                value=v.value,
                nvalue=v.nvalue,
                idx_data_points=None,
                idx_split_variable=v.idx_split_variable,
                linear_params=v.linear_params,
            )
            for k, v in self.tree_structure.items()
        }
        return Tree(
            tree_structure=tree,
            idx_leaf_nodes=None,
            output=np.array([-1]),
            split_rules=self.split_rules,
        )

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
        self,
        x: npt.NDArray[np.float_],
        excluded: Optional[List[int]] = None,
        shape: int = 1,
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

        return self._traverse_tree(X=x, excluded=excluded, shape=shape)

    def _traverse_tree(
        self,
        X: npt.NDArray[np.float_],
        excluded: Optional[List[int]] = None,
        shape: Union[int, Tuple[int, ...]] = 1,
    ) -> npt.NDArray[np.float_]:
        """
        Traverse the tree starting from the root node given an (un)observed point.

        Parameters
        ----------
        X : npt.NDArray[np.float_]
            (Un)observed point(s)
        node_index : int
            Index of the node to start the traversal from
        split_variable : int
            Index of the variable used to split the node
        excluded: Optional[List[int]]
            Indexes of the variables to exclude when computing predictions

        Returns
        -------
        npt.NDArray[np.float_]
            Leaf node value or mean of leaf node values
        """

        x_shape = (1,) if len(X.shape) == 1 else X.shape[:-1]
        nd_dims = (...,) + (None,) * len(x_shape)

        stack = [(0, np.ones(x_shape), 0)]  # (node_index, weight, idx_split_variable) initial state
        p_d = (
            np.zeros(shape + x_shape) if isinstance(shape, tuple) else np.zeros((shape,) + x_shape)
        )
        while stack:
            node_index, weights, idx_split_variable = stack.pop()
            node = self.get_node(node_index)
            if node.is_leaf_node():
                params = node.linear_params
                if params is None:
                    p_d += weights * node.value[nd_dims]
                else:
                    p_d += weights * (
                        params[0][nd_dims] + params[1][nd_dims] * X[..., idx_split_variable]
                    )
            else:
                idx_split_variable = node.idx_split_variable
                left_node_index, right_node_index = get_idx_left_child(
                    node_index
                ), get_idx_right_child(node_index)
                if excluded is not None and idx_split_variable in excluded:
                    prop_nvalue_left = self.get_node(left_node_index).nvalue / node.nvalue
                    stack.append((left_node_index, weights * prop_nvalue_left, idx_split_variable))
                    stack.append(
                        (right_node_index, weights * (1 - prop_nvalue_left), idx_split_variable)
                    )
                else:
                    to_left = (
                        self.split_rules[idx_split_variable]
                        .divide(X[..., idx_split_variable], node.value)
                        .astype("float")
                    )
                    stack.append((left_node_index, weights * to_left, idx_split_variable))
                    stack.append((right_node_index, weights * (1 - to_left), idx_split_variable))

        if len(X.shape) == 1:
            p_d = p_d[..., 0]

        return p_d

    def _traverse_leaf_values(
        self, leaf_values: List[npt.NDArray[np.float_]], leaf_n_values: List[int], node_index: int
    ) -> None:
        """
        Traverse the tree appending leaf values starting from a particular node.

        Parameters
        ----------
        leaf_values : List[npt.NDArray[np.float_]]
        node_index : int
        """
        node = self.get_node(node_index)
        if node.is_leaf_node():
            leaf_values.append(node.value)
            leaf_n_values.append(node.nvalue)
        else:
            self._traverse_leaf_values(leaf_values, leaf_n_values, get_idx_left_child(node_index))
            self._traverse_leaf_values(leaf_values, leaf_n_values, get_idx_right_child(node_index))
