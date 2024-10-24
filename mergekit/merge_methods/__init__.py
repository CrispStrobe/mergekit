# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.


from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
    SparsificationMethod,
    get_enhanced_ties_params,
)
from mergekit.merge_methods.linear import LinearMerge
from mergekit.merge_methods.model_stock import ModelStockMerge
from mergekit.merge_methods.passthrough import PassthroughMerge
from mergekit.merge_methods.slerp import SlerpMerge
from mergekit.merge_methods.tokenizer_permute import TokenizerPermutationMerge


def get(method: str) -> MergeMethod:
    if method == "linear":
        return LinearMerge()
    elif method == "slerp":
        return SlerpMerge()
    elif method == "passthrough":
        return PassthroughMerge()
    elif method == "task_arithmetic":
        return GeneralizedTaskArithmeticMerge(
            consensus=None,  # Corrected parameter name
            sparsification=None,  # Corrected parameter name
            default_normalize=False,
            default_rescale=False,
        )
    elif method == "ties":
        return GeneralizedTaskArithmeticMerge(
            consensus=ConditionalParameter(value=ConsensusMethod.sum.value),  # Corrected parameter name and wrapped in ConditionalParameter
            sparsification=ConditionalParameter(value=SparsificationMethod.magnitude.value),  # Corrected parameter name and wrapped
            default_normalize=True,
            default_rescale=False,
        )
    elif method == "ties_enhanced":
        params = get_enhanced_ties_params()
        return GeneralizedTaskArithmeticMerge(
            consensus=params["consensus"],  # Corrected parameter name
            sparsification=params["sparsification"],  # Corrected parameter name
            default_normalize=params["default_normalize"],
            default_rescale=params["default_rescale"],
        )
    elif method == "dare_ties":
        return GeneralizedTaskArithmeticMerge(
            consensus=ConditionalParameter(value=ConsensusMethod.sum.value),
            sparsification=ConditionalParameter(value=SparsificationMethod.random.value),
            default_normalize=False,
            default_rescale=True,
        )
    elif method == "dare_linear":
        return GeneralizedTaskArithmeticMerge(
            consensus=None,
            sparsification=ConditionalParameter(value=SparsificationMethod.random.value),
            default_normalize=False,
            default_rescale=True,
        )
    elif method == "breadcrumbs":
        return GeneralizedTaskArithmeticMerge(
            consensus=None,
            sparsification=ConditionalParameter(value=SparsificationMethod.magnitude_outliers.value),
            default_normalize=False,
            default_rescale=False,
        )
    elif method == "breadcrumbs_ties":
        return GeneralizedTaskArithmeticMerge(
            consensus=ConditionalParameter(value=ConsensusMethod.sum.value),
            sparsification=ConditionalParameter(value=SparsificationMethod.magnitude_outliers.value),
            default_normalize=False,
            default_rescale=False,
        )
    elif method == "model_stock":
        return ModelStockMerge()
    elif method == "della":
        return GeneralizedTaskArithmeticMerge(
            consensus=ConditionalParameter(value=ConsensusMethod.sum.value),
            sparsification=ConditionalParameter(value=SparsificationMethod.rank_magnitude_sampling.value),
            default_normalize=True,
            default_rescale=True,
        )
    elif method == "della_linear":
        return GeneralizedTaskArithmeticMerge(
            consensus=None,
            sparsification=ConditionalParameter(value=SparsificationMethod.rank_magnitude_sampling.value),
            default_normalize=False,
            default_rescale=True,
        )
    raise RuntimeError(f"Unimplemented merge method {method}")

__all__ = [
    "MergeMethod",
    "get",
    "LinearMerge",
    "SlerpMerge",
    "PassthroughMerge",
    "GeneralizedTaskArithmeticMerge",
    "TokenizerPermutationMerge",
]
