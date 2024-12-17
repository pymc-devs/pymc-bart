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
import pymc as pm

from pymc_bart.bart import BART
from pymc_bart.pgbart import PGBART
from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule, SubsetSplitRule
from pymc_bart.utils import (
    compute_variable_importance,
    plot_convergence,
    plot_dependence,
    plot_ice,
    plot_pdp,
    plot_scatter_submodels,
    plot_variable_importance,
    plot_variable_inclusion,
)

__all__ = [
    "BART",
    "PGBART",
    "ContinuousSplitRule",
    "OneHotSplitRule",
    "SubsetSplitRule",
    "compute_variable_importance",
    "plot_convergence",
    "plot_dependence",
    "plot_ice",
    "plot_pdp",
    "plot_scatter_submodels",
    "plot_variable_importance",
    "plot_variable_inclusion",
]
__version__ = "0.8.0"


pm.STEP_METHODS = list(pm.STEP_METHODS) + [PGBART]
