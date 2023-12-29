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

from typing import List, Optional, Tuple, Union
import numpy.typing as npt
import numpy as np
from numba import njit
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor import config
from pytensor import function as pytensor_function
from pytensor.tensor.var import Variable

from pymc_bart.bart import BARTRV
from pymc_bart.tree import Node, Tree, get_idx_left_child, get_idx_right_child, get_depth
from pymc_bart.split_rules import ContinuousSplitRule


class ParticleTree:
    """Particle tree."""

    __slots__ = "tree", "expansion_nodes", "log_weight"

    def __init__(self, tree: Tree):
        self.tree: Tree = tree.copy()
        self.expansion_nodes: List[int] = [0]
        self.log_weight: float = 0

    def copy(self) -> "ParticleTree":
        p = ParticleTree(self.tree)
        p.expansion_nodes = self.expansion_nodes.copy()
        return p

    def sample_tree(
        self,
        ssv,
        available_predictors,
        prior_prob_leaf_node,
        X,
        missing_data,
        sum_trees,
        leaf_sd,
        m,
        response,
        normal,
        shape,
    ) -> bool:
        tree_grew = False
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            prob_leaf = prior_prob_leaf_node[get_depth(index_leaf_node)]

            if prob_leaf < np.random.random():
                idx_new_nodes = grow_tree(
                    self.tree,
                    index_leaf_node,
                    ssv,
                    available_predictors,
                    X,
                    missing_data,
                    sum_trees,
                    leaf_sd,
                    m,
                    response,
                    normal,
                    shape,
                )
                if idx_new_nodes is not None:
                    self.expansion_nodes.extend(idx_new_nodes)
                    tree_grew = True

        return tree_grew


class PGBART(ArrayStepShared):
    """
    Particle Gibss BART sampling step.

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : tuple
        Number of particles. Defaults to 10
    batch : tuple
        Number of trees fitted per step. The first element is the batch size during tuning and the
        second the batch size after tuning.  Defaults to  (0.1, 0.1), meaning 10% of the `m` trees
        during tuning and after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    name = "pgbart"
    default_blocked = False
    generates_stats = True
    stats_dtypes = [{"variable_inclusion": object, "tune": bool}]

    def __init__(
        self,
        vars=None,  # pylint: disable=redefined-builtin
        num_particles: int = 10,
        batch: Tuple[float, float] = (0.1, 0.1),
        model: Optional[Model] = None,
    ):
        model = modelcontext(model)
        initial_values = model.initial_point()
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)
        value_bart = vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        if isinstance(self.bart.X, Variable):
            self.X = self.bart.X.eval()
        else:
            self.X = self.bart.X

        self.missing_data = np.any(np.isnan(self.X))
        self.m = self.bart.m
        self.response = self.bart.response

        shape = initial_values[value_bart.name].shape

        self.shape = 1 if len(shape) == 1 else shape[0]

        # Set trees_shape (dim for separate tree structures)
        # and leaves_shape (dim for leaf node values)
        # One of the two is always one, the other equal to self.shape
        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1

        if self.bart.split_prior.size == 0:
            self.alpha_vec = np.ones(self.X.shape[1])
        else:
            self.alpha_vec = self.bart.split_prior

        if self.bart.split_rules:
            self.split_rules = self.bart.split_rules
        else:
            self.split_rules = [ContinuousSplitRule] * self.X.shape[1]

        for idx, rule in enumerate(self.split_rules):
            if rule is ContinuousSplitRule:
                self.X[:, idx] = jitter_duplicated(self.X[:, idx], np.nanstd(self.X[:, idx]))

        init_mean = self.bart.Y.mean()
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))

        # if data is binary
        self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape))

        y_unique = np.unique(self.bart.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            self.leaf_sd *= 3 / self.m**0.5
        else:
            self.leaf_sd *= self.bart.Y.std() / self.m**0.5

        self.running_sd = [
            RunningSd((self.leaves_shape, self.num_observations)) for _ in range(self.trees_shape)
        ]

        self.sum_trees = np.full(
            (self.trees_shape, self.leaves_shape, self.bart.Y.shape[0]), init_mean
        ).astype(config.floatX)
        self.sum_trees_noi = self.sum_trees - init_mean
        self.a_tree = Tree.new_tree(
            leaf_node_value=init_mean / self.m,
            idx_data_points=np.arange(self.num_observations, dtype="int32"),
            num_observations=self.num_observations,
            shape=self.leaves_shape,
            split_rules=self.split_rules,
        )

        self.normal = NormalSampler(1, self.leaves_shape)
        self.uniform = UniformSampler(0, 1)
        self.prior_prob_leaf_node = compute_prior_probability(self.bart.alpha, self.bart.beta)
        self.ssv = SampleSplittingVariable(self.alpha_vec)

        self.tune = True

        batch_0 = max(1, int(self.m * batch[0]))
        batch_1 = max(1, int(self.m * batch[1]))
        self.batch = (batch_0, batch_1)

        self.num_particles = num_particles
        self.indices = list(range(1, num_particles))
        shared = make_shared_replacements(initial_values, vars, model)
        self.likelihood_logp = logp(initial_values, [model.datalogp], vars, shared)
        self.all_particles = [
            [ParticleTree(self.a_tree) for _ in range(self.m)] for _ in range(self.trees_shape)
        ]
        self.all_trees = np.array([[p.tree for p in pl] for pl in self.all_particles])
        self.lower = 0
        self.iter = 0
        super().__init__(vars, shared)

    def astep(self, _):
        variable_inclusion = np.zeros(self.num_variates, dtype="int")

        upper = min(self.lower + self.batch[~self.tune], self.m)
        tree_ids = range(self.lower, upper)
        self.lower = upper if upper < self.m else 0

        for odim in range(self.trees_shape):
            for tree_id in tree_ids:
                self.iter += 1
                # Compute the sum of trees without the old tree that we are attempting to replace
                self.sum_trees_noi[odim] = (
                    self.sum_trees[odim] - self.all_particles[odim][tree_id].tree._predict()
                )
                # Generate an initial set of particles
                # at the end we return one of these particles as the new tree
                particles = self.init_particles(tree_id, odim)

                while True:
                    # Sample each particle (try to grow each tree), except for the first one
                    stop_growing = True
                    for p in particles[1:]:
                        if p.sample_tree(
                            self.ssv,
                            self.available_predictors,
                            self.prior_prob_leaf_node,
                            self.X,
                            self.missing_data,
                            self.sum_trees[odim],
                            self.leaf_sd[odim],
                            self.m,
                            self.response,
                            self.normal,
                            self.leaves_shape,
                        ):
                            self.update_weight(p, odim)
                        if p.expansion_nodes:
                            stop_growing = False
                    if stop_growing:
                        break

                    # Normalize weights
                    normalized_weights = self.normalize(particles[1:])

                    # Resample
                    particles = self.resample(particles, normalized_weights)

                normalized_weights = self.normalize(particles)
                # Get the new particle and associated tree
                self.all_particles[odim][tree_id], new_tree = self.get_particle_tree(
                    particles, normalized_weights
                )
                # Update the sum of trees
                new = new_tree._predict()
                self.sum_trees[odim] = self.sum_trees_noi[odim] + new
                # To reduce memory usage, we trim the tree
                self.all_trees[odim][tree_id] = new_tree.trim()

                if self.tune:
                    # Update the splitting variable and the splitting variable sampler
                    if self.iter > self.m:
                        self.ssv = SampleSplittingVariable(self.alpha_vec)

                    for index in new_tree.get_split_variables():
                        self.alpha_vec[index] += 1

                    # update standard deviation at leaf nodes
                    if self.iter > 2:
                        self.leaf_sd[odim] = self.running_sd[odim].update(new)
                    else:
                        self.running_sd[odim].update(new)

                else:
                    # update the variable inclusion
                    for index in new_tree.get_split_variables():
                        variable_inclusion[index] += 1

        if not self.tune:
            self.bart.all_trees.append(self.all_trees)

        stats = {"variable_inclusion": variable_inclusion, "tune": self.tune}
        return self.sum_trees, [stats]

    def normalize(self, particles: List[ParticleTree]) -> float:
        """
        Use softmax to get normalized_weights.
        """
        log_w = np.array([p.log_weight for p in particles])
        log_w_max = log_w.max()
        log_w_ = log_w - log_w_max
        wei = np.exp(log_w_) + 1e-12
        return wei / wei.sum()

    def resample(
        self, particles: List[ParticleTree], normalized_weights: npt.NDArray[np.float_]
    ) -> List[ParticleTree]:
        """
        Use systematic resample for all but the first particle

        Ensure particles are copied only if needed.
        """
        new_indices = self.systematic(normalized_weights) + 1
        seen: List[int] = []
        new_particles: List[ParticleTree] = []
        for idx in new_indices:
            if idx in seen:
                new_particles.append(particles[idx].copy())
            else:
                new_particles.append(particles[idx])
                seen.append(idx)

        particles[1:] = new_particles

        return particles

    def get_particle_tree(
        self, particles: List[ParticleTree], normalized_weights: npt.NDArray[np.float_]
    ) -> Tuple[ParticleTree, Tree]:
        """
        Sample a new particle and associated tree
        """
        new_index = self.systematic(normalized_weights)[
            discrete_uniform_sampler(self.num_particles)
        ]
        new_particle = particles[new_index]

        return new_particle, new_particle.tree

    def systematic(self, normalized_weights: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
        """
        Systematic resampling.

        Return indices in the range 0, ..., len(normalized_weights)

        Note: adapted from https://github.com/nchopin/particles
        """
        lnw = len(normalized_weights)
        single_uniform = (self.uniform.rvs() + np.arange(lnw)) / lnw
        return inverse_cdf(single_uniform, normalized_weights)

    def init_particles(self, tree_id: int, odim: int) -> List[ParticleTree]:
        """Initialize particles."""
        p0: ParticleTree = self.all_particles[odim][tree_id]
        # The old tree does not grow so we update the weight only once
        self.update_weight(p0, odim)
        particles: List[ParticleTree] = [p0]

        particles.extend(ParticleTree(self.a_tree) for _ in self.indices)
        return particles

    def update_weight(self, particle: ParticleTree, odim: int) -> None:
        """
        Update the weight of a particle.
        """

        delta = (
            np.identity(self.trees_shape)[odim][:, None, None]
            * particle.tree._predict()[None, :, :]
        )

        new_likelihood = self.likelihood_logp((self.sum_trees_noi + delta).flatten())
        particle.log_weight = new_likelihood

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


class RunningSd:
    """Welford's online algorithm for computing the variance/standard deviation"""

    def __init__(self, shape: tuple) -> None:
        self.count = 0  # number of data points
        self.mean = np.zeros(shape)  # running mean
        self.m_2 = np.zeros(shape)  # running second moment

    def update(self, new_value: npt.NDArray[np.float_]) -> Union[float, npt.NDArray[np.float_]]:
        self.count = self.count + 1
        self.mean, self.m_2, std = _update(self.count, self.mean, self.m_2, new_value)
        return fast_mean(std)


@njit
def _update(
    count: int,
    mean: npt.NDArray[np.float_],
    m_2: npt.NDArray[np.float_],
    new_value: npt.NDArray[np.float_],
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], Union[float, npt.NDArray[np.float_]]]:
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m_2 += delta * delta2

    std = (m_2 / count) ** 0.5
    return mean, m_2, std


class SampleSplittingVariable:
    def __init__(self, alpha_vec: npt.NDArray[np.float_]) -> None:
        """
        Sample splitting variables proportional to `alpha_vec`.

        This is equivalent to compute the posterior mean of a Dirichlet-Multinomial model.
        This enforce sparsity.
        """
        self.enu = list(enumerate(np.cumsum(alpha_vec / alpha_vec.sum())))

    def rvs(self) -> Union[int, Tuple[int, float]]:
        rnd: float = np.random.random()
        for i, val in self.enu:
            if rnd <= val:
                return i
        return self.enu[-1]


def compute_prior_probability(alpha: int, beta: int) -> List[float]:
    """
    Calculate the probability of the node being a leaf node (1 - p(being split node)).

    Parameters
    ----------
    alpha : float
    beta: float

    Returns
    -------
    list with probabilities for leaf nodes
    """
    prior_leaf_prob: List[float] = [0]
    depth = 0
    while prior_leaf_prob[-1] < 0.9999:
        prior_leaf_prob.append(1 - (alpha * ((1 + depth) ** (-beta))))
        depth += 1
    prior_leaf_prob.append(1)

    return prior_leaf_prob


def grow_tree(
    tree,
    index_leaf_node,
    ssv,
    available_predictors,
    X,
    missing_data,
    sum_trees,
    leaf_sd,
    m,
    response,
    normal,
    shape,
):
    current_node = tree.get_node(index_leaf_node)
    idx_data_points = current_node.idx_data_points

    index_selected_predictor = ssv.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    idx_data_points, available_splitting_values = filter_missing_values(
        X[idx_data_points, selected_predictor], idx_data_points, missing_data
    )

    split_rule = tree.split_rules[selected_predictor]

    split_value = split_rule.get_split_value(available_splitting_values)

    if split_value is None:
        return None

    to_left = split_rule.divide(available_splitting_values, split_value)
    new_idx_data_points = idx_data_points[to_left], idx_data_points[~to_left]

    current_node_children = (
        get_idx_left_child(index_leaf_node),
        get_idx_right_child(index_leaf_node),
    )

    if response == "mix":
        response = "linear" if np.random.random() >= 0.5 else "constant"

    for idx in range(2):
        idx_data_point = new_idx_data_points[idx]
        node_value, linear_params = draw_leaf_value(
            y_mu_pred=sum_trees[:, idx_data_point],
            x_mu=X[idx_data_point, selected_predictor],
            m=m,
            norm=normal.rvs() * leaf_sd,
            shape=shape,
            response=response,
        )

        new_node = Node.new_leaf_node(
            value=node_value,
            nvalue=len(idx_data_point),
            idx_data_points=idx_data_point,
            linear_params=linear_params,
        )
        tree.set_node(current_node_children[idx], new_node)

    tree.grow_leaf_node(current_node, selected_predictor, split_value, index_leaf_node)
    return current_node_children


def filter_missing_values(available_splitting_values, idx_data_points, missing_data):
    if missing_data:
        mask = ~np.isnan(available_splitting_values)
        idx_data_points = idx_data_points[mask]
        available_splitting_values = available_splitting_values[mask]
    return idx_data_points, available_splitting_values


def draw_leaf_value(
    y_mu_pred: npt.NDArray[np.float_],
    x_mu: npt.NDArray[np.float_],
    m: int,
    norm: npt.NDArray[np.float_],
    shape: int,
    response: str,
) -> Tuple[npt.NDArray[np.float_], Optional[npt.NDArray[np.float_]]]:
    """Draw Gaussian distributed leaf values."""
    linear_params = None
    mu_mean = np.empty(shape)
    if y_mu_pred.size == 0:
        return np.zeros(shape), linear_params

    if y_mu_pred.size == 1:
        mu_mean = np.full(shape, y_mu_pred.item() / m) + norm
    else:
        if y_mu_pred.size < 3 or response == "constant":
            mu_mean = fast_mean(y_mu_pred) / m + norm
        else:
            mu_mean, linear_params = fast_linear_fit(x=x_mu, y=y_mu_pred, m=m, norm=norm)

    return mu_mean, linear_params


@njit
def fast_mean(ari: npt.NDArray[np.float_]) -> Union[float, npt.NDArray[np.float_]]:
    """Use Numba to speed up the computation of the mean."""
    if ari.ndim == 1:
        count = ari.shape[0]
        suma = 0
        for i in range(count):
            suma += ari[i]
        return suma / count
    else:
        res = np.zeros(ari.shape[0])
        count = ari.shape[1]
        for j in range(ari.shape[0]):
            for i in range(count):
                res[j] += ari[j, i]
        return res / count


@njit
def fast_linear_fit(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    m: int,
    norm: npt.NDArray[np.float_],
) -> Tuple[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]]:
    n = len(x)
    y = y / m + np.expand_dims(norm, axis=1)

    xbar = np.sum(x) / n
    ybar = np.sum(y, axis=1) / n

    x_diff = x - xbar
    y_diff = y - np.expand_dims(ybar, axis=1)

    x_var = np.dot(x_diff, x_diff.T)

    if x_var == 0:
        b = np.zeros(y.shape[0])
    else:
        b = np.dot(x_diff, y_diff.T) / x_var

    a = ybar - b * xbar

    y_fit = np.expand_dims(a, axis=1) + np.expand_dims(b, axis=1) * x
    return y_fit.T, [a, b]


def discrete_uniform_sampler(upper_value):
    """Draw from the uniform distribution with bounds [0, upper_value).

    This is the same and np.random.randit(upper_value) but faster.
    """
    return int(np.random.random() * upper_value)


class NormalSampler:
    """Cache samples from a standard normal distribution."""

    def __init__(self, scale, shape):
        self.size = 1000
        self.scale = scale
        self.shape = shape
        self.update()

    def rvs(self):
        if self.idx == self.size:
            self.update()
        pop = self.cache[:, self.idx]
        self.idx += 1
        return pop

    def update(self):
        self.idx = 0
        self.cache = np.random.normal(loc=0.0, scale=self.scale, size=(self.shape, self.size))


class UniformSampler:
    """Cache samples from a uniform distribution."""

    def __init__(self, lower_bound, upper_bound, shape=None):
        self.size = 1000
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.shape = shape
        self.update()

    def rvs(self):
        if self.idx == self.size:
            self.update()
        if self.shape is None:
            pop = self.cache[self.idx]
        else:
            pop = self.cache[:, self.idx]
        self.idx += 1
        return pop

    def update(self):
        self.idx = 0
        if self.shape is None:
            self.cache = np.random.uniform(self.lower_bound, self.upper_bound, size=self.size)
        else:
            self.cache = np.random.uniform(
                self.lower_bound, self.upper_bound, size=(self.shape, self.size)
            )


@njit
def inverse_cdf(
    single_uniform: npt.NDArray[np.float_], normalized_weights: npt.NDArray[np.float_]
) -> npt.NDArray[np.int_]:
    """
    Inverse CDF algorithm for a finite distribution.

    Parameters
    ----------
    single_uniform: npt.NDArray[np.float_]
        Ordered points in [0,1]

    normalized_weights: npt.NDArray[np.float_])
        Normalized weights

    Returns
    -------
    new_indices: ndarray
        a vector of indices in range 0, ..., len(normalized_weights)

    Note: adapted from https://github.com/nchopin/particles
    """
    idx = 0
    a_weight = normalized_weights[0]
    sul = len(single_uniform)
    new_indices = np.empty(sul, dtype=np.int64)
    for i in range(sul):
        while single_uniform[i] > a_weight:
            idx += 1
            a_weight += normalized_weights[idx]
        new_indices[i] = idx
    return new_indices


@njit
def jitter_duplicated(array: npt.NDArray[np.float_], std: float) -> npt.NDArray[np.float_]:
    """
    Jitter duplicated values.
    """
    if are_whole_number(array):
        seen = []
        for idx, num in enumerate(array):
            if num in seen and not np.isnan(num):
                array[idx] = num + np.random.normal(0, std / 12)
            else:
                seen.append(num)

    return array


@njit
def are_whole_number(array: npt.NDArray[np.float_]) -> np.bool_:
    """Check if all values in array are whole numbers"""
    return np.all(np.mod(array[~np.isnan(array)], 1) == 0)


def logp(point, out_vars, vars, shared):  # pylint: disable=redefined-builtin
    """Compile PyTensor function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc.Distribution` for the output variables
    vars: List
        containing :class:`pymc.Distribution` for the input variables
    shared: List
        containing :class:`pytensor.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    function = pytensor_function([inarray0], out_list[0])
    function.trust_input = True
    return function
