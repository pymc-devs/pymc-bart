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

from numba import njit
import numpy as np

class ContinuousSplitRule:
    """
    Standard continuous split rule: pick a pivot value and split depending on if variable is smaller or greater than the value picked.
    """

    @staticmethod
    @njit
    def decide(value, split_value):
        return value <= split_value

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1:
            idx_selected_splitting_values = int(np.random.random() * len(available_splitting_values))
            split_value = available_splitting_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def get_new_idx_data_points(available_splitting_values, split_value, idx_data_points):
        split_idx = available_splitting_values <= split_value
        return idx_data_points[split_idx], idx_data_points[~split_idx]

class OneHotSplitRule:
    """Choose a single categorical value and branch on if the variable is that value or not"""

    @staticmethod
    @njit
    def decide(value, split_value):
        return value == split_value

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1 and not np.all(available_splitting_values==available_splitting_values[0]):
            idx_selected_splitting_values = int(np.random.random() * len(available_splitting_values))
            split_value = available_splitting_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def get_new_idx_data_points(available_splitting_values, split_value, idx_data_points):
        split_idx = available_splitting_values == split_value
        return idx_data_points[split_idx], idx_data_points[~split_idx]


class SubsetSplitRule:
    """
    Choose a random subset of the categorical values and branch on if the value is within the chosen set.
    This is the approach taken in flexBART paper.
    """

    @staticmethod
    def decide(value, split_value):
        return (value in split_value)

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1 and not np.all(available_splitting_values==available_splitting_values[0]):
            unique_values = np.unique(available_splitting_values)[:-1] # Remove last one so it always goes to right
            while True:
                sample = np.random.randint(0, 2, size=len(unique_values)).astype(bool)
                if np.any(sample): break
            split_value = unique_values[sample]
        return split_value

    @staticmethod
    def get_new_idx_data_points(available_splitting_values, split_value, idx_data_points):
        split_idx = np.isin(available_splitting_values, split_value)
        return idx_data_points[split_idx], idx_data_points[~split_idx]
