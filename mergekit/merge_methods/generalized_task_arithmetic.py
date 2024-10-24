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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Literal

# Import ConditionalParameter from config.py
from mergekit.config import ConditionalParameter
from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.sparsify import SparsificationMethod, sparsify


# Define ConsensusMethod Enum
class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


def get_enhanced_ties_params() -> Dict[str, Any]:
    """
    Returns default parameters for enhanced TIES.
    Now returns strings instead of ConditionalParameter.
    """
    return {
        "consensus": ConsensusMethod.sum,  # Direct Enum value
        "sparsification": SparsificationMethod.magnitude,  # Direct Enum value
        "default_normalize": True,
        "default_rescale": False,
        "model_parameters": {  # Add default model parameters
            "weight": 1.1,
            "density": 1.0,
            "epsilon": 0.95,
            "lambda": 1.5
        }
    }

class GeneralizedTaskArithmeticMerge(MergeMethod, BaseModel):
    """
    A Pydantic model representing the configuration for the Generalized Task Arithmetic Merge method.
    """
    consensus_method: Optional[ConsensusMethod] = Field(  # Changed to direct Enum
        default=None, description="Consensus method for merging."
    )
    sparsification_method: Optional[SparsificationMethod] = Field(  # Changed to direct Enum
        default=None, description="Sparsification method for merging."
    )
    default_normalize: bool = Field(
        default=True, description="Default normalization flag."
    )
    default_rescale: bool = Field(
        default=False, description="Default rescale flag."
    )

    class Config:
        frozen = True

    def get_enhanced_ties_params() -> Dict[str, Any]:
        """
        Returns default parameters for enhanced TIES.
        Now returns strings instead of ConditionalParameter.
        """
        return {
            "consensus": ConsensusMethod.sum,  # Direct Enum value
            "sparsification": SparsificationMethod.magnitude,  # Direct Enum value
            "default_normalize": True,
            "default_rescale": False,
            "model_parameters": {  # Add default model parameters
                "weight": 1.1,
                "density": 1.0,
                "epsilon": 0.95,
                "lambda": 1.5
            }
        }

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> "GTATask":
        print("DEBUG: MAKE_TASK INPUT")
        print(f"parameters: {parameters}")
        print(f"parameters data: {parameters.data}")
    
        # Create tensor parameters with default values
        default_params = get_enhanced_ties_params()["model_parameters"]
        
        tensor_params_dict = {}
        for model_ref in tensor_parameters.data.keys():
            if model_ref != base_model:  # Only add parameters for non-base models
                tensor_params_dict[model_ref] = default_params.copy()
                print(f"Injected default params for {model_ref}: {tensor_params_dict[model_ref]}")
            else:
                tensor_params_dict[model_ref] = {}
    
        processed_tensor_parameters = ImmutableMap(tensor_params_dict)
    
        return GTATask(
            method=self,  # Changed from self.__class__(...) to just self
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=processed_tensor_parameters,
            int8_mask=parameters["int8_mask"] if "int8_mask" in parameters else False,
            normalize=parameters["normalize"] if "normalize" in parameters else self.default_normalize,
            rescale=parameters["rescale"] if "rescale" in parameters else self.default_rescale,
            post_process_factor=parameters["post_process_factor"] if "post_process_factor" in parameters else 1.0,
            magnitude_threshold=parameters["magnitude_threshold"] if "magnitude_threshold" in parameters else 1e-4,
            weight_info=output_weight,
        )


class GTATask(Task[torch.Tensor]):
    """
    Represents a Generalized Task Arithmetic (GTA) Task for merging tensors.
    """

    method: GeneralizedTaskArithmeticMerge
    tensors: MergeTensorInput
    base_model: ModelReference
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool
    rescale: bool
    post_process_factor: float
    magnitude_threshold: float
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        """
        Indicates whether this task utilizes hardware accelerators.
        """
        return True

    def arguments(self) -> Dict[str, Task]:
        """
        Defines the arguments required to execute this task.
        """
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
    
        if self.method.sparsification_method:  # Using sparsification_method
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
                    method=self.method.sparsification_method,  # Using sparsification_method
                    rescale=self.rescale,
                    **kwargs,
                )
    
        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        weights = torch.tensor(
            [tv["weight"] for tv in tvs],
            dtype=deltas.dtype,
            device=deltas.device,
        )
        while len(deltas.shape) > len(weights.shape):
            weights = weights.unsqueeze(-1)
    
        weighted_deltas = deltas * weights
    
        # Changed from consensus to consensus_method
        if self.method.consensus_method:
            mask_dtype = torch.int8 if self.int8_mask else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method.value,  # Using consensus_method.value
                mask_dtype=mask_dtype,
                magnitude_threshold=self.magnitude_threshold,
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

    

    def group_label(self) -> Optional[str]:
        """
        Returns the group label for this task, if any.
        """
        return self.tensors.group_label()


def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, Dict[str, Any]],
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

        # Debug print
        print(f"Model: {model}")
        print(f"Tensor parameters: {tensor_parameters[model]}")  # Changed this line

        d = {
            "model": model,
            "delta": delta
        }
        # Add all parameters from tensor_parameters
        if model in tensor_parameters:
            params = tensor_parameters[model]
            if hasattr(params, 'data'):  # If it's an ImmutableMap
                params = params.data
            for key, value in params.items():
                d[key] = value
                print(f"Adding parameter {key}: {value}")

        res.append(d)
        print(f"Final d: {d.keys()}")

    return res, base

def execute(
    self,
    tensors: Dict[ModelReference, torch.Tensor],
    **_kwargs,
) -> torch.Tensor:
    # Add debug prints
    print("Starting execute")
    tvs, base = get_task_vectors(
        self.weight_info,
        self.base_model,
        tensors,
        tensor_parameters=self.tensor_parameters.data,
    )
    print(f"Task vectors: {[tv.keys() for tv in tvs]}")
    if not tvs:
        return base


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
    magnitude_threshold: float = 1e-4,
) -> torch.Tensor:
    """
    Generates a mask determining which delta vectors should be merged into the final model.

    Args:
        delta (torch.Tensor): The tensor of deltas to be merged.
        method (Literal["sum", "count"], optional): The consensus method to use. Defaults to "sum".
        mask_dtype (Optional[torch.dtype], optional): The data type of the mask. Defaults to None.
        magnitude_threshold (float, optional): The threshold for magnitude-based masking. Defaults to 1e-4.

    Returns:
        torch.Tensor: The generated mask tensor.
    """
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

