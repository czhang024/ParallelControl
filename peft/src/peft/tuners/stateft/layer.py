# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from accelerate.utils.imports import is_xpu_available

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.tuners.lora.dora import DoraLinearLayer
from peft.utils.other import transpose



class StateFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("stateft",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.input_shape = None
        self.output_shape = None
        self.stateft = nn.ModuleDict()
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.kwargs = kwargs

        base_layer = self.get_base_layer()

    def update_layer(self, adapter_name, *args, **kwargs):
        raise NotImplementedError(
            f"update_layer is not implemented for {self.__class__.__name__}. "
            "Please implement it in the derived class."
        )

    def base_update_layer(self, adapter_name, adapter_layer, init_weights):
        assert isinstance(adapter_layer, nn.Module), (
            f"adapter_layer should be an instance of nn.Module, but got {type(adapter_layer)}"
        )   
        self.stateft[adapter_name] = adapter_layer

        if init_weights:
            self.reset_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_parameters(self, adapter_name):
        raise NotImplementedError(
            f"reset_parameters is not implemented for {self.__class__.__name__}. "
            "Please implement it in the derived class."
        )
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.stateft.keys():
                    continue
                result = result + self.stateft[active_adapter](x, *args, **kwargs).to(result.dtype)

        result = result.to(previous_dtype)
        return result


class LoRAsideLayer(nn.Module):
    """
    A side layer for LoRA-like layers that contains the LoRA parameters.
    This is used to separate the LoRA parameters from the base layer parameters.
    """

    def __init__(self, in_features: int, out_features: int, r: int,
            lora_alpha: float = 1.0, lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            lora_bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.update_layer(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
        )
    def update_layer(
        self,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout=lora_dropout_layer
        # Actual trainable parameters
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias = lora_bias

        if use_rslora:
            self.scaling = lora_alpha / math.sqrt(r)
        else:
            self.scaling = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        
        if init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B.weight)
        elif init_lora_weights:
            self.reset_lora_parameters(init_lora_weights)

    def reset_lora_parameters(self, init_lora_weights):
        if init_lora_weights is False:
            return

        if init_lora_weights is True:
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.r)
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_bias:
            nn.init.zeros_(self.lora_B.bias)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B.weight.device
        dtype = self.lora_B.weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A.weight
        weight_B = self.lora_B.weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A.weight.data = weight_A.to(dtype)
            self.lora_B.weight.data = weight_B.to(dtype)

        return output_tensor
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        lora_A = self.lora_A
        lora_B = self.lora_B
        dropout = self.lora_dropout
        scaling = self.scaling
        result = lora_B(lora_A(dropout(x.to(lora_A.weight.dtype)))) * scaling

        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "StateFT." + rep

class StateFTLoraLayer(nn.Module,StateFTLayer):
    # StateFT implemented as a LoRA-like layer

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        in_features,
        out_features,
        r,
        lora_alpha,
        lora_dropout,
        init_weights,
        use_rslora,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        assert not isinstance(base_layer, nn.Embedding), 'StateFTLora does not support embedding layers.' 
        super().__init__()
        StateFTLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name

        self.init_weights = init_weights
        self.kwargs = kwargs

        self.update_layer(
            adapter_name, 
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_weights=init_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self,
        adapter_name,
        in_features,
        out_features,
        r,
        lora_alpha,
        lora_dropout,
        init_weights,
        use_rslora,
        lora_bias: bool = False,):
        # This code works for linear layers, override for other layer types
        lora_layer= LoRAsideLayer(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
        )
        self.base_update_layer(adapter_name, lora_layer, False)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        raise ValueError("StateFTLora does not support merging")

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        raise ValueError("StateFTLora does not support merging")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "StateFTLora." + rep

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return StateFTLayer.forward(self, x, *args, **kwargs)

    @torch.no_grad()
    def reset_parameters(self, adapter_name):
        """
        Reset the parameters of the adapter layer.
        """
        if adapter_name not in self.stateft:
            raise ValueError(f"Adapter {adapter_name} not found in {self.__class__.__name__}")

        init_lora_weights = self.init_weights if self.init_weights else True
        self.stateft[adapter_name].reset_lora_parameters(init_lora_weights)
