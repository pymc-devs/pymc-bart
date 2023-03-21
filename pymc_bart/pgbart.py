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

from typing import List, Optional, Union

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
from pymc_bart.tree import Node, Tree, get_depth


class ParticleTree:
    """Particle tree."""

    __slots__ = "tree", "expansion_nodes", "log_weight", "kfactor"

    def __init__(self, tree: Tree, kfactor: float = 0.75):
        self.tree: Tree = tree.copy()
        self.expansion_nodes: List[int] = [0]
        self.log_weight: float = 0
        self.kfactor: float = kfactor

    def copy(self) -> "ParticleTree":
        p = ParticleTree(self.tree)
        p.expansion_nodes = self.expansion_nodes.copy()
        p.kfactor = self.kfactor
        return p

    def sample_tree(
        self,
        ssv,
        available_predictors,
        prior_prob_leaf_node,
        X,
        missing_data,
        sum_trees,
        m,
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
                    m,
                    normal,
                    self.kfactor,
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
        Number of particles. Defaults to 20
    batch : int or tuple
        Number of trees fitted per step. Defaults to  "auto", which is the 10% of the `m` trees
        during tuning and after tuning. If a tuple is passed the first element is the batch size
        during tuning and the second the batch size after tuning.
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
        num_particles: int = 20,
        batch: Union[str, int] = "auto",
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
        shape = initial_values[value_bart.name].shape
        if len(shape) == 1:
            self.shape = 1
        else:
            self.shape = shape[0]

        if self.bart.split_prior:
            self.alpha_vec = self.bart.split_prior
        else:
            self.alpha_vec = np.ones(self.X.shape[1], dtype=np.int32)
        init_mean = self.bart.Y.mean()
        # if data is binary
        y_unique = np.unique(self.bart.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            mu_std = 3 / self.m**0.5
        else:
            mu_std = self.bart.Y.std() / self.m**0.5

        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))

        self.sum_trees = np.full((self.shape, self.bart.Y.shape[0]), init_mean).astype(
            config.floatX
        )
        self.sum_trees_noi = self.sum_trees - init_mean
        self.a_tree = Tree.new_tree(
            leaf_node_value=init_mean / self.m,
            idx_data_points=np.arange(self.num_observations, dtype="int32"),
            num_observations=self.num_observations,
            shape=self.shape,
        )
        self.normal = NormalSampler(mu_std, self.shape)
        self.uniform = UniformSampler(0, 1)
        self.uniform_kf = UniformSampler(0.33, 0.75, self.shape)
        self.prior_prob_leaf_node = compute_prior_probability(self.bart.alpha)
        self.ssv = SampleSplittingVariable(self.alpha_vec)

        self.tune = True

        if batch == "auto":
            batch = max(1, int(self.m * 0.1))
            self.batch = (batch, batch)
        else:
            if isinstance(batch, (tuple, list)):
                self.batch = batch
            else:
                self.batch = (batch, batch)

        self.num_particles = num_particles
        self.indices = list(range(1, num_particles))
        shared = make_shared_replacements(initial_values, vars, model)
        self.likelihood_logp = logp(initial_values, [model.datalogp], vars, shared)
        self.all_particles = list(ParticleTree(self.a_tree) for _ in range(self.m))
        self.all_trees = np.array([p.tree for p in self.all_particles])
        self.lower = 0
        self.iter = 0
        super().__init__(vars, shared)

    def astep(self, _):
        variable_inclusion = np.zeros(self.num_variates, dtype="int")

        upper = min(self.lower + self.batch[~self.tune], self.m)
        tree_ids = range(self.lower, upper)
        self.lower = upper if upper < self.m else 0

        for tree_id in tree_ids:
            self.iter += 1
            # Compute the sum of trees without the old tree that we are attempting to replace
            self.sum_trees_noi = self.sum_trees - self.all_particles[tree_id].tree._predict()
            # Generate an initial set of particles
            # at the end we return one of these particles as the new tree
            particles = self.init_particles(tree_id)

            while True:
                # Sample each particle (try to grow each tree), except for the first one
                stop_growing = True
                for p in particles[1:]:
                    tree_grew = p.sample_tree(
                        self.ssv,
                        self.available_predictors,
                        self.prior_prob_leaf_node,
                        self.X,
                        self.missing_data,
                        self.sum_trees,
                        self.m,
                        self.normal,
                        self.shape,
                    )
                    if tree_grew:
                        self.update_weight(p)
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
            self.all_particles[tree_id], new_tree = self.get_particle_tree(
                particles, normalized_weights
            )
            # Update the sum of trees
            self.sum_trees = self.sum_trees_noi + new_tree._predict()
            # To reduce memory usage, we trim the tree
            self.all_trees[tree_id] = new_tree.trim()

            if self.tune:
                # Update the splitting variable and the splitting variable sampler
                if self.iter > self.m:
                    self.ssv = SampleSplittingVariable(self.alpha_vec)
                for index in new_tree.get_split_variables():
                    self.alpha_vec[index] += 1
            else:
                # update the variable inclusion
                for index in new_tree.get_split_variables():
                    variable_inclusion[index] += 1

        if not self.tune:
            self.bart.all_trees.append(self.all_trees)

        stats = {"variable_inclusion": variable_inclusion, "tune": self.tune}
        return self.sum_trees, [stats]

    def normalize(self, particles) -> float:
        """
        Use softmax to get normalized_weights.
        """
        log_w = np.array([p.log_weight for p in particles])
        log_w_max = log_w.max()
        log_w_ = log_w - log_w_max
        wei = np.exp(log_w_) + 1e-12
        return wei / wei.sum()

    def resample(self, particles: List[ParticleTree], normalized_weights) -> List[ParticleTree]:
        """
        Use systematic resample for all but the first particle

        Ensure particles are copied only if needed.
        """
        new_indices = self.systematic(normalized_weights) + 1
        seen = []
        new_particles = []
        for idx in new_indices:
            if idx in seen:
                new_particles.append(particles[idx].copy())
            else:
                new_particles.append(particles[idx])
                seen.append(idx)

        particles[1:] = new_particles

        return particles

    def get_particle_tree(self, particles, normalized_weights):
        """
        Sample a new particle and associated tree
        """
        new_index = self.systematic(normalized_weights)[
            discrete_uniform_sampler(self.num_particles)
        ]
        new_particle = particles[new_index]

        return new_particle, new_particle.tree

    def systematic(self, normalized_weights):
        """
        Systematic resampling.

        Return indices in the range 0, ..., len(normalized_weights)

        Note: adapted from https://github.com/nchopin/particles
        """
        lnw = len(normalized_weights)
        single_uniform = (self.uniform.rvs() + np.arange(lnw)) / lnw
        return inverse_cdf(single_uniform, normalized_weights)

    def init_particles(self, tree_id: int) -> List[ParticleTree]:
        """Initialize particles."""
        p0: ParticleTree = self.all_particles[tree_id]
        # The old tree does not grow so we update the weight only once
        self.update_weight(p0)
        particles: List[ParticleTree] = [p0]

        particles.extend(
            ParticleTree(self.a_tree, self.uniform_kf.rvs() if self.tune else p0.kfactor)
            for _ in self.indices
        )
        return particles

    def update_weight(self, particle):
        """
        Update the weight of a particle.
        """
        new_likelihood = self.likelihood_logp(
            (self.sum_trees_noi + particle.tree._predict()).flatten()
        )
        particle.log_weight = new_likelihood

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


class SampleSplittingVariable:
    def __init__(self, alpha_vec):
        """
        Sample splitting variables proportional to `alpha_vec`.

        This is equivalent to compute the posterior mean of a Dirichlet-Multinomial model.
        This enforce sparsity.
        """
        self.enu = list(enumerate(np.cumsum(alpha_vec / alpha_vec.sum())))

    def rvs(self):
        rnd = np.random.random()
        for i, val in self.enu:
            if rnd <= val:
                return i
        return self.enu[-1]


def compute_prior_probability(alpha):
    """
    Calculate the probability of the node being a leaf node (1 - p(being split node)).

    Taken from equation 19 in [Rockova2018].

    Parameters
    ----------
    alpha : float

    Returns
    -------
    list with probabilities for leaf nodes

    References
    ----------
    .. [Rockova2018] Veronika Rockova, Enakshi Saha (2018). On the theory of BART.
    arXiv, `link <https://arxiv.org/abs/1810.00787>`__
    """
    prior_leaf_prob = [0]
    depth = 1
    while prior_leaf_prob[-1] < 1:
        prior_leaf_prob.append(1 - alpha**depth)
        depth += 1
    return prior_leaf_prob


def grow_tree(
    tree,
    index_leaf_node,
    ssv,
    available_predictors,
    X,
    missing_data,
    sum_trees,
    m,
    normal,
    kfactor,
    shape,
):
    current_node = tree.get_node(index_leaf_node)
    idx_data_points = current_node.idx_data_points

    index_selected_predictor = ssv.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    available_splitting_values = X[idx_data_points, selected_predictor]
    split_value = get_split_value(available_splitting_values, idx_data_points, missing_data)

    if split_value is None:
        return None
    new_idx_data_points = get_new_idx_data_points(
        available_splitting_values, split_value, idx_data_points
    )
    current_node_children = (
        current_node.get_idx_left_child(),
        current_node.get_idx_right_child(),
    )

    new_nodes = np.array([])
    for idx in range(2):
        idx_data_point = new_idx_data_points[idx]
        node_value = draw_leaf_value(
            sum_trees[:, idx_data_point],
            m,
            normal.rvs() * kfactor,
            shape,
        )

        new_node = Node.new_leaf_node(
            index=current_node_children[idx],
            value=node_value,
            idx_data_points=idx_data_point,
        )
        new_nodes = np.append(new_nodes, new_node)

    tree.grow_leaf_node(current_node, selected_predictor, split_value, index_leaf_node)
    tree.set_node(new_nodes[0].index, new_nodes[0])
    tree.set_node(new_nodes[1].index, new_nodes[1])

    return [new_nodes[0].index, new_nodes[1].index]


@njit
def get_new_idx_data_points(available_splitting_values, split_value, idx_data_points):
    split_idx = available_splitting_values <= split_value
    return idx_data_points[split_idx], idx_data_points[~split_idx]


def get_split_value(available_splitting_values, idx_data_points, missing_data):
    if missing_data:
        idx_data_points = idx_data_points[~np.isnan(available_splitting_values)]
        available_splitting_values = available_splitting_values[
            ~np.isnan(available_splitting_values)
        ]

    split_value = None
    if available_splitting_values.size > 0:
        idx_selected_splitting_values = discrete_uniform_sampler(len(available_splitting_values))
        split_value = available_splitting_values[idx_selected_splitting_values]

    return split_value


@njit
def draw_leaf_value(y_mu_pred, m, norm, shape):
    """Draw Gaussian distributed leaf values."""
    if y_mu_pred.size == 0:
        return np.zeros(shape)

    if y_mu_pred.size == 1:
        mu_mean = np.full(shape, y_mu_pred.item() / m)
    else:
        mu_mean = fast_mean(y_mu_pred) / m

    return norm + mu_mean


@njit
def fast_mean(ari):
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
def inverse_cdf(single_uniform, normalized_weights):
    """
    Inverse CDF algorithm for a finite distribution.

    Parameters
    ----------
    single_uniform: ndarray
        ordered points in [0,1]

    normalized_weights: ndarray
        normalized weights

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
