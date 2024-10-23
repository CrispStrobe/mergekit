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

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.sparsify import SparsificationMethod, sparsify

class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"

def get_enhanced_ties_params():
    return {
        "consensus": ConsensusMethod.sum,
        "sparsification": SparsificationMethod.rank_magnitude_sampling,
        "default_normalize": False,
        "post_process_factor": 1.5,
        "magnitude_threshold": 1e-4,
        "epsilon": 0.95,
        "lambda": 1.5
    }

class GeneralizedTaskArithmeticMerge(MergeMethod, BaseModel, frozen=True):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    default_rescale: bool

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
            ConfigParameterDef(
                name="rescale", required=False, default_value=self.default_rescale
            ),
            ConfigParameterDef(
                name="post_process_factor", required=False, default_value=1.0
            ),
            ConfigParameterDef(
                name="magnitude_threshold", required=False, default_value=1e-4
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        res = [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
        ]
        if self.sparsification_method == SparsificationMethod.magnitude_outliers:
            res.append(
                ConfigParameterDef(
                    name="gamma",
                    default_value=0.01,
                )
            )
        if self.sparsification_method == SparsificationMethod.rank_magnitude_sampling:
            res.append(
                ConfigParameterDef(
                    name="epsilon",
                    default_value=0.95,  # Will be overridden by global param if set
                )
            )
            res.append(
                ConfigParameterDef(
                    name="lambda",
                    default_value=1.5,  # Will be overridden by global param if set
                )
            )
        return res

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        # Use dictionary access for ImmutableMap
        consensus_method = self.consensus_method
        if "consensus_method" in parameters:
            consensus_method = ConsensusMethod[parameters["consensus_method"]]
            
        sparsification_method = self.sparsification_method
        if "sparsification_method" in parameters:
            sparsification_method = (
                SparsificationMethod[parameters["sparsification_method"]]
                if parameters["sparsification_method"]
                else None
            )
    
        # Propagate epsilon and lambda to tensor parameters if provided
        if "epsilon" in parameters or "lambda" in parameters:
            new_tensor_parameters = ImmutableMap(tensor_parameters.data.copy())
            for model_ref in new_tensor_parameters.keys():
                params = dict(new_tensor_parameters[model_ref])
                if "epsilon" in parameters:
                    params["epsilon"] = parameters["epsilon"]
                if "lambda" in parameters:
                    params["lambda"] = parameters["lambda"]
                new_tensor_parameters = new_tensor_parameters.set(model_ref, ImmutableMap(params))
        else:
            new_tensor_parameters = tensor_parameters
    
        return GTATask(
            method=self.__class__(
                consensus_method=consensus_method,
                sparsification_method=sparsification_method,
                default_normalize=self.default_normalize,
                default_rescale=self.default_rescale,
            ),
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=new_tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize=parameters["normalize"],
            rescale=parameters["rescale"],
            post_process_factor=parameters["post_process_factor"] if "post_process_factor" in parameters else 1.0,
            magnitude_threshold=parameters["magnitude_threshold"] if "magnitude_threshold" in parameters else 1e-4,
            weight_info=output_weight,
        )

class GTATask(Task[torch.Tensor]):
    method: GeneralizedTaskArithmeticMerge
    tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool
    rescale: bool
    post_process_factor: float
    magnitude_threshold: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        tvs, base = get_task_vectors(
            self.weight_info,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters.data,
        )
        if not tvs:
            return base

        if self.method.sparsification_method:
            for tv_info in tvs:
                kwargs = {}
                if "gamma" in tv_info:
                    kwargs["gamma"] = tv_info["gamma"]
                if "epsilon" in tv_info:
                    kwargs["epsilon"] = tv_info["epsilon"]
                if "lambda" in tv_info:
                    kwargs["lambda"] = tv_info["lambda"]
                    
                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info["density"],
                    method=self.method.sparsification_method,
                    rescale=self.rescale,
                    **kwargs,
                )

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        if self.method.consensus_method:
            mask_dtype = torch.int8 if self.int8_mask else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method,
                mask_dtype=mask_dtype,
                magnitude_threshold=self.magnitude_threshold
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        if self.normalize:
            mixed_delta /= divisor

        mixed_delta *= self.post_process_factor

        return (base + mixed_delta).to(base.dtype)

def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]
    parameter_name = weight_info.name

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].to(base.dtype)
        if x.shape != base.shape:
            if weight_info.is_embed:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(f"skipping {model}:{parameter_name} due to size mismatch")
                continue

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in tensor_parameters[model]:
            d[p] = tensor_parameters[model][p]
        res.append(d)
    return res, base

def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
    magnitude_threshold: float = 1e-4,
):
    if mask_dtype is None:
        mask_dtype = delta.dtype

    magnitude_mask = delta.abs() > magnitude_threshold
    sign = delta.sign().to(mask_dtype)
    sign = sign * magnitude_mask.to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign
