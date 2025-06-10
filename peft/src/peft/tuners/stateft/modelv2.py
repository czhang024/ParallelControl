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
from __future__ import annotations

import re
import warnings
from dataclasses import asdict
from enum import Enum
from itertools import chain
from typing import Optional, Any, Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists, _find_minimal_target_modules
from peft.utils import (
    # TRANSFORMERS_MODELS_TO_STATEFT_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)
from peft.utils.other import get_pattern_key

from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING

from peft.utils.constants import (
    DUMMY_MODEL_CONFIG,
    DUMMY_TARGET_MODULES,
    EMBEDDING_LAYER_NAMES,
    MIN_TARGET_MODULES_FOR_OPTIMIZATION,
    SEQ_CLS_HEAD_NAMES,
)
from peft.utils.integrations import init_empty_weights
from peft.utils.peft_types import PeftType, TaskType

from .config import StateFTv2Config, StateFTLorav2Config
from .layer import LoRAsideLayer

from collections import defaultdict

import inspect
from copy import deepcopy
from functools import update_wrapper
from types import MethodType



def _remove_dag_hooks(model, edge_name=None, adapter_name: str = None):
    """
    Remove dag hooks from the model.
    """
    if not hasattr(model, 'has_dag_hooks') or not model.has_dag_hooks:
        assert not hasattr(model, 'dag_hook_handles') or len(model.dag_hook_handles) == 0, \
            "Model has DAG hooks but 'has_dag_hooks' is not set. Please check the model."
        warnings.warn("Model does not have DAG hooks. No hooks to remove.")
        return 
    
    if edge_name is not None:
        if edge_name not in model.dag_hook_handles:
            warnings.warn(f"Edge {edge_name} not found in model dag_hook_handles. No hooks to remove.")
            return
        handles = model.dag_hook_handles[edge_name]
        if adapter_name is not None:
            if adapter_name not in handles:
                warnings.warn(f"Adapter {adapter_name} not found in edge {edge_name}. No hooks to remove.")
                return
            handles = handles.pop(adapter_name,None)
            if isinstance(handles, (tuple, list)):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()
        else:
            for handle in handles:
                if isinstance(handle, (tuple, list)):
                    for h in handle:
                        h.remove()
                else:
                    handle.remove()
            del model.dag_hook_handles[edge_name]
        if len(model.dag_hook_handles) == 0:
            model.has_dag_hooks = False
    else:
        for edge_name, handles in model.dag_hook_handles.items():
            if adapter_name is None:
                for handle in handles:
                    if isinstance(handle, (tuple, list)):
                        for handle in handles:
                            handle.remove()
                    else:
                        handle.remove()
            elif adapter_name in handles:
                handles = handles.pop(adapter_name)
                if handles is None:
                    continue
                if isinstance(handles[adapter_name], (tuple, list)):
                    for handle in handles[adapter_name]:
                        handle.remove()
                else:
                    handles[adapter_name].remove()
        if adapter_name is None:
            model.dag_hook_handles = defaultdict(dict)
        model.has_dag_hooks = False if adapter_name is None or len(model.dag_hook_handles) == 0 else True

class BaseDAGControlModel(BaseTuner):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("shortcut_module",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ()

    prefix: str = "stateftv2_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False, **kwargs) -> None:
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.kwargs = kwargs
        self._edges=kwargs.get('edges', None)
        self._insert_nodes=kwargs.get('insert_nodes', None)
        
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        
        # from peft.helpers import update_signature
        # update_signature(self)

    ########## BaseTuner ##########
    def _check_new_adapter_config(self, config: StateFTv2Config, edges: Dict=None, insert_nodes: Dict=None) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # if config.in_features is None or config.out_features is None:
        #     raise ValueError(
        #         f"Please specify `in_features` and `out_features` in the config. "
        #         f"Got in_features: {config.in_features}, out_features: {config.out_features}."
        #     )
        # if config.in_features <= 0:
        #     raise ValueError(f"in_features should be greater than 0, got {config.in_features}.")
        # if config.out_features <= 0:
        #     raise ValueError(f"out_features should be greater than 0, got {config.out_features}.")
        #TODO: check if the target modules exist in the model
        assert edges is not None or insert_nodes is not None, "Either edges or insert_nodes must be provided."
        submodules= dict(self.model.named_modules())
        if edges is not None:
            for (head, tail), adapter_module in edges.items():
                assert head in submodules, f"Head {head} not found in model submodules."
                assert tail in submodules, f"Tail {tail} not found in model submodules."
                assert isinstance(adapter_module, nn.Module), f"Submodule {adapter_module} is not a valid nn.Module."
        if insert_nodes is not None:
            for name, adapter_module in insert_nodes.items():
                assert name in submodules, f"Insert node {name} not found in model submodules."
                assert isinstance(adapter_module, nn.Module), f"Submodule {adapter_module} is not a valid nn.Module."

    @staticmethod
    def _check_target_module_exists(StateFTv2_config, key):
        return True

    def _create_and_replace(
        self,
        StateFT_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        warnings.warn('This implementation does not use `_create_and_replace` method. ')
        return 
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `_create_and_replace` method. "
            "This method should be implemented in the child class."
        )
    
    def _replace_module(self, parent, child_name, new_module, child):
        """
        This implementation using hook to achieve adapter and does not need to replace the module 
        except for the modules to save and unloading.
        """
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        if hasattr(child, 'device'):
            device= child.device
        else:
            for p in child.parameters():
                device = p.device
                break
        
        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer
    
    def inject_adapter(
        self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False
    ) -> None:
        peft_config = self.peft_config[adapter_name]
        edges = self._edges
        insert_nodes = self._insert_nodes
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config, edges=edges, insert_nodes=insert_nodes)

        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False

        model_config = self.get_model_config(model)

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)
        key_list = [key for key, _ in model.named_modules()]

        uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        if uses_dummy_target_modules:
            # dummy adapter, we allow not matching any module
            key_list = []

        # This is an optimization to reduce the number of entries in the target_modules list. The reason is that in some
        # circumstances, target_modules can contain hundreds of entries. Since each target module is checked against
        # each module of the net (which can be thousands), this can become quite expensive when many adapters are being
        # added. Often, the target_modules can be condensed in such a case, which speeds up the process.
        # A context in which this can happen is when diffusers loads non-PEFT LoRAs. As there is no meta info on
        # target_modules in that case, they are just inferred by listing all keys from the state_dict, which can be
        # quite a lot. See: https://github.com/huggingface/diffusers/issues/9297
        # As there is a small chance for undiscovered bugs, we apply this optimization only if the list of
        # target_modules is sufficiently big.
        # We also exclude IA³ from this optimization. This is because IA³ has both target_modules and
        # feedforward_modules, which are coupled (the latter must be a subset). It would be possible to change the logic
        # to keep both in sync, but it's not quite trivial and probably not worth the effort. See #2429.
        if (
            isinstance(peft_config.target_modules, (list, set))
            and (len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION)
            and (peft_config.peft_type != PeftType.IA3)
        ):
            names_no_target = [
                name
                for name in key_list
                if not any((name == suffix) or name.endswith("." + suffix) for suffix in peft_config.target_modules)
            ]
            new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
            if len(new_target_modules) < len(peft_config.target_modules):
                peft_config.target_modules = new_target_modules

        for key in key_list:
            if not key:
                continue
            # Check for modules_to_save in case
            #
            # Note that this is redundant with PeftModel.set_additional_trainable_models but might be necessary
            # when calling inject_adapter without a PEFT model. This is outdated as it only focuses on
            # ModulesToSaveWrapper and ignores other potentially configured AuxiliaryTrainingWrapper instances.
            #
            # TODO: determine if there's a good reason for this and refactor to support AuxiliaryTrainingWrapper,
            # or remove if superfluous.
            if _check_for_modules_to_save and any(
                key.endswith(module_to_save) for module_to_save in peft_config.modules_to_save
            ):
                # Optionally set the modules to save
                parent, target, target_name = _get_submodules(model, key)

                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True

        self.build_edges(
            edges=edges,
            insert_nodes=insert_nodes,
            adapter_name=adapter_name,
            )
        # del self._edges, self._insert_nodes
        self.set_adapter(self.active_adapters)
        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, "modules_to_save"):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if 'shortcut_module' not in n:
                p.requires_grad = False
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        self.enable_apaters(enabled)
        self._disable_adapters = not enabled
        for module in self.model.modules():
            if isinstance(module, (BaseDAGControlModel, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_apaters(self, enabled: bool = True) -> None:
        """Toggle the enabling and disabling of adapters

        Args:
            enabled (`bool`, *optional*, defaults to `True`): Whether to enable or disable the adapters.
        """
        if enabled:
            self.set_adapter(self.active_adapter)
        else:
            self.model.remove_dag_hooks()

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for edge_name, adapter_modules in self.shortcut_modules.items():
            for _adapter_name,module in adapter_modules.items():
                if _adapter_name in adapter_names:
                    self.register_dag_hook(self.get_edge(edge_name), module, adapter_name=_adapter_name)

        for module in self.model.modules():
            if isinstance(module, BaseDAGControlModel):
                module.set_adapter(adapter_names)
        self.active_adapter = adapter_names

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
            # if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_STATEFT_TARGET_MODULES_MAPPING:
            #     raise ValueError("Please specify `target_modules` in `peft_config`")
            # peft_config.target_modules = set(
            #     TRANSFORMERS_MODELS_TO_STATEFT_TARGET_MODULES_MAPPING[model_config["model_type"]]
            # )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=False,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        assert not merge, 'Merging is not supported for StateFTModel.'
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        del self.shortcut_modules
        del self.shortcut_states
        if hasattr(self.model, 'dag_hook_handles'):
            self.model.remove_dag_hooks()
            del self.model.dag_hook_handles
        return self.model

    def delete_adapter(self, adapter_name: str=None, edge_name=None) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if edge_name is None:
            if adapter_name not in list(self.peft_config.keys()):
                warnings.warn(f"Adapter {adapter_name} does not exist")
            del self.peft_config[adapter_name]

            for attr in self.adapter_layer_names + self.other_param_names:
                if adapter_name in getattr(self, attr):
                    del getattr(self, attr)[adapter_name]

            # we cannot use self.prefix as we want to include non-trainable StateFT parameters
            new_adapter = None
            self.model.remove_dag_hooks(adapter_name=adapter_name)
            for edge_name, adapter_modules in self.shortcut_modules.items():
                if adapter_name in adapter_modules:
                    del adapter_modules[adapter_name]
                if new_adapter is None:
                    new_adapter = list(adapter_modules.keys())

            self.active_adapter = new_adapter or []
        else:
            if edge_name not in self.model.dag_hook_handles:
                warnings.warn(f"Edge {edge_name} does not exist")
            if adapter_name not in self.model.dag_hook_handles[edge_name]:
                warnings.warn(f"Adapter {adapter_name} does not exist in edge {edge_name}")
            self.model.remove_dag_hooks(edge_name=edge_name, adapter_name=adapter_name)
            del self.shortcut_modules[edge_name][adapter_name]
            if len(self.shortcut_modules[edge_name]) == 0:
                del self.shortcut_modules[edge_name]

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        Merging is not supported for StateFTModel. 
        """
        raise ValueError(
            "Merging is not supported for StateFTModel. "
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the StateFT modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
    
    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, edge=None, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        assert edge is not None or device is not None, "Either edge or device must be provided."
        if device is None:
            base_module = self.model.get_submodule(edge[1] if isinstance(edge, tuple) else edge)
            for p in base_module.parameters():
                device = p.device
                break
    
        meta = torch.device("meta")

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        edge_name = self.get_edge_name(edge)
        if adapter_name not in self.shortcut_modules[edge_name]:
            raise ValueError(f"Adapter {adapter_name} not found in edge {edge}.")
        if not any(p.device == meta for p in self.shortcut_modules[edge_name][adapter_name].parameters()):
            if p.dtype.is_floating_point or p.dtype.is_complex:
                self.shortcut_modules[edge_name][adapter_name] = self.shortcut_modules[edge_name][adapter_name].to(device, dtype=p.dtype)
            else:
                self.shortcut_modules[edge_name][adapter_name] = self.shortcut_modules[edge_name][adapter_name].to(device)

        # for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
        #     adapter_layer = getattr(self, adapter_layer_name, None)
        #     if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict)) or adapter_layer_name == 'shortcut_module':
        #         continue
        #     if adapter_name not in adapter_layer:
        #         continue
        #     if any(p.device == meta for p in adapter_layer.parameters()):
        #         continue

        #     if p.dtype.is_floating_point or p.dtype.is_complex:
        #         adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=p.dtype)
        #     else:
        #         adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    ####### BaseDAGControlModel ########
    def get_edge_name(self, edge: Tuple[str, str]|str) -> str:
        """
        Get the name of the edge in the DAG.

        Args:
            edge (tuple|str): A tuple of (head, tail) representing the edge in the DAG or a string representing the node name.
        
        Returns:
            str: The name of the edge.
        """
        if isinstance(edge, tuple):
            head = edge[0].replace('.', '-')
            tail = edge[1].replace('.', '-')
            return '-TO-'.join([head, tail])
        else:
            return edge.replace('.', '-')
    def get_edge(self, edge_name: str) -> Tuple[str, str]|str:
        """
        Get the edge from the DAG by its name.

        Args:
            edge_name (str): The name of the edge in the DAG.
        
        Returns:
            tuple|str: A tuple of (head, tail) representing the edge in the DAG or a string representing the node name.
        """
        if '-TO-' in edge_name:
            head, tail = edge_name.split('-TO-')
            return (head.replace('-', '.'), tail.replace('-', '.'))
        else:
            return edge_name.replace('-', '.')

    def build_edges(self, edges: Dict, insert_nodes: Dict = None, adapter_name: str = 'default'):
        """
        Build the additional edges in the DAG.

        Args:
            edges (dict): A dictionary representing the additional edges in the DAG. 
                          The keys (head,tail) are tuples of submodule names of the additional edges,
                            and the values are the corresponding submodule instances.
            insert_nodes (dict): A dictionary representing the additional nodes in the DAG. 
                                The keys are submodule names after which the additional nodes will be inserted,
                                and the values are the corresponding submodule instances.
            adapter_name (str): The name of the adapter to be used for the additional edges and nodes.
        """
        if hasattr(self.model, 'has_dag_hooks') and self.model.has_dag_hooks:
            raise ValueError("Model already has DAG hooks. Please remove them before adding new ones.")
        # self.edges = edges
        submodules = dict(self.model.named_modules())
        if not hasattr(self.model, 'dag_hook_handles'):
            self.model.dag_hook_handles = defaultdict(dict)
        if not hasattr(self, 'shortcut_modules'):
            self.shortcut_modules = nn.ModuleDict()
        if not hasattr(self, 'shortcut_states'):
            self.shortcut_states = {}
        if edges is not None:
            for edge, submodule in edges.items():
                self.register_dag_hook(edge, submodule, adapter_name=adapter_name)
                self._move_adapter_to_device_of_base_layer(adapter_name, edge)
                
        if insert_nodes is not None:
            for name, submodule in insert_nodes.items():
                self.register_dag_hook(name, submodule, adapter_name=adapter_name)
                self._move_adapter_to_device_of_base_layer(adapter_name, name)
        # self.shortcut_modules=nn.ModuleDict({k: v for k,v in self.shortcut_modules.items()})
        self.model.has_dag_hooks = True
        self.model.remove_dag_hooks = MethodType(_remove_dag_hooks, self.model)

    def _head_hook_fn(self, submodule, tail):
        """
        Create a forward hook function for the head of an edge.

        Args:
            submodule: The submodule to attach the hook to.

        Returns:
            A function that takes the module, input, and output and applies the submodule.
        """
        def hook(module, input):
            self.shortcut_states[tail].append(submodule(*input))
        return hook
    
    def _tail_hook_fn(self, submodule, tail):
        """
        Create a forward hook function for the tail of an edge.

        Args:
            submodule: The submodule to attach the hook to.

        Returns:
            A function that takes the module, input, and output and applies the submodule.
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                outputx = output[0]
            else:
                outputx = output
            for states in self.shortcut_states[tail]:
                assert states.shape == outputx.shape, f"Shape mismatch: {states.shape} vs {outputx.shape}, for tail {tail}"
                outputx = states + outputx.to(states.device).to(states.dtype)
            self.shortcut_states[tail] = []
            if isinstance(output, tuple):
                output = (outputx,) + output[1:]
            else:
                output = outputx
            return output
        return hook
    
    def _insert_node_hook_fn(self, submodule):
        """
        Create a forward hook function for an insert node.

        Args:
            submodule: The submodule to attach the hook to.

        Returns:
            A function that takes the module, input, and output and applies the submodule.
        """
        def hook(module, input, output):
            return submodule(output).to(output.device).to(output.dtype)
        return hook

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass through the model.
        Args:
            inputs: The inputs to the model.
        Returns:
            The output of the model.
        """
        return self.model.forward(*args, **kwargs)
    
    def delete_all_dag_hooks(self):
        """
        Remove all hooks from the model.
        """
        self.model.remove_dag_hooks()
        self.shortcut_states = {}
        if hasattr(self, 'shortcut_modules'):
            del self.shortcut_modules

    def register_dag_hook(self, edge, adapter_module: nn.Module, adapter_name: str = 'default'):
        """
        Register a new DAG hook.

        Args:
            edge (tuple|str): A tuple of (head, tail) representing the edge in the DAG.
            adapter_module (nn.Module): The adapter_module to be added as a hook.
            adapter_name (str): The name of the adapter.
        """
        if not hasattr(self.model, 'dag_hook_handles') or not hasattr(self, 'shortcut_modules'):
            raise ValueError("Model does not have DAG hooks. Please build the edges first.")
        adapter_module.to(self.model.device)
        if isinstance(edge, tuple):
            head = edge[0].replace('.', '-')
            tail = edge[1].replace('.', '-')
            edge_name = '-TO-'.join([head, tail])
            if edge_name in self.shortcut_modules and adapter_name in self.shortcut_modules[edge_name]:
                self.model.remove_dag_hooks(edge_name=edge_name, adapter_name=adapter_name)
            inhook = self.model.get_submodule(edge[0]).register_forward_pre_hook(self._head_hook_fn(adapter_module, tail))
            outhook = self.model.get_submodule(edge[1]).register_forward_hook(self._tail_hook_fn(adapter_module, tail))
            if edge_name not in self.shortcut_modules:
                self.shortcut_modules[edge_name] = nn.ModuleDict()
            self.shortcut_modules[edge_name][adapter_name] = adapter_module
            self.model.dag_hook_handles[edge_name][adapter_name] = (inhook,outhook)
            if tail not in self.shortcut_states:
                self.shortcut_states[tail] = []
        else:
            nodehook=self.model.get_submodule(edge).register_forward_hook(self._insert_node_hook_fn(adapter_module))
            name = edge.replace('.', '-')
            if name in self.shortcut_modules and adapter_name in self.shortcut_modules[name]:
                self.model.remove_dag_hooks(edge_name=name, adapter_name=adapter_name)
            if name not in self.shortcut_modules:
                self.shortcut_modules[name] = nn.ModuleDict()
            self.shortcut_modules[name].append(adapter_module)
            self.model.dag_hook_handles[name][adapter_name]=nodehook
            if name not in self.shortcut_states:
                self.shortcut_states[name] = []
        

class ParallelControlModel(BaseDAGControlModel):
    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False, edges: Dict = None):
        """
        Initialize the ParallelControlModel class.

        Args:
            model (nn.Module): The neural network model to control.
            target_modules (list): List of target modules to apply Low-Rank side control to.
            r (int): Rank of the LoRA layers.
            control_alpha (int): Scaling factor for the LoRA layers.
            lora_dropout (float): Dropout probability for the LoRA layers.
        """
        if edges is None:
            edges = self.create_lora_edges(model, config[adapter_name])
        super(ParallelControlModel, self).__init__(model, config, adapter_name, low_cpu_mem_usage, edges=edges)

    def _check_new_adapter_config(self, config: StateFTLorav2Config, edges: Dict=None, insert_nodes: Dict=None) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config, edges=edges, insert_nodes=insert_nodes)
        if config.lora_alpha <= 0:
            raise ValueError(f"lora_alpha should be greater than 0, got {config.lora_alpha}.")
        if config.r <= 0:
            raise ValueError(f"r should be greater than 0, got {config.r}.")
        if not isinstance(config.target_modules, (list, dict, set)):
            raise ValueError(
                f"target_modules should be a list/set or a dict, got {type(config.target_modules)}."
                "Please specify target_modules as a list/set of strings (target module) or a dict of strings (target module with specified (in_features, out_features))."
            )
        if (isinstance(config.target_modules, (list, set)) and not all(isinstance(target, str) for target in config.target_modules)) or \
           (isinstance(config.target_modules, dict) and not all(isinstance(target, str) for target in config.target_modules.keys())):
            raise ValueError(
                f"target_modules should be a list/set of strings (target module) or a dict of strings (target module with specified (in_features, out_features)), got {config.target_modules}."
            )
        if isinstance(config.target_modules, dict) and not all(
            isinstance(features, (tuple,list)) and len(features) == 2 for features in config.target_modules.values()):
            raise ValueError(
                f"the values of target_modules should be a 2-tuple (in_features, out_features), got {config.target_modules}.")

    def create_lora_edges(self, model: nn.Module, config: StateFTLorav2Config,
                          ):
        """
        Create LoRA edges for the model.

        Args:
            model (nn.Module): The neural network model to control.
            config (StateFTLorav2Config): Configuration for the LoRA layers, including target modules, rank, alpha, dropout, etc.

        Returns:
            dict: A dictionary representing the additional edges in the DAG.
        """
        edges = {}
        submodules = dict(model.named_modules())
        target_modules = config.target_modules
        for current_key in submodules.keys():
            if isinstance(target_modules, list):
                if any(current_key.endswith(target) for target in target_modules):
                    edges[(current_key, current_key)] =  LoRAsideLayer(
                        in_features=config.in_features,
                        out_features=config.out_features,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                        init_lora_weights=config.init_weights,
                        use_rslora=config.use_rslora,
                        lora_bias=config.lora_bias,
                    )
            elif isinstance(target_modules, dict):
                for target, (in_features, out_features) in target_modules.items():
                    if current_key.endswith(target):
                        edges[(current_key, current_key)] =  LoRAsideLayer(
                            in_features=in_features,
                            out_features=out_features,
                            r=config.r,
                            lora_alpha=config.lora_alpha,
                            lora_dropout=config.lora_dropout,
                            init_lora_weights=config.init_weights,
                            use_rslora=config.use_rslora,
                            lora_bias=config.lora_bias,
                        )
        return edges
        
class ParallelControlv2Model(ParallelControlModel):
    prefix: str = "stateft_lora_v2_"
    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        """
        """
        edges = self.create_lora_edges(model, config[adapter_name])
        super(ParallelControlModel, self).__init__(model, config, adapter_name, low_cpu_mem_usage, edges=edges)
    def _check_new_adapter_config(self, config: StateFTLorav2Config, edges: Dict=None, insert_nodes: Dict=None) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config, edges=edges, insert_nodes=insert_nodes)
        if (isinstance(config.target_modules, (list, set)) and not all(isinstance(target, str) or (isinstance(target, tuple) and len(target)==2) for target in config.target_modules)) or \
           (isinstance(config.target_modules, dict) and not all(isinstance(target, str) or (isinstance(target, tuple) and len(target)==2) for target in config.target_modules.keys())):
            raise ValueError(
                f"target_modules should be a list of strings (target_module) and 2-tuple (head, tail target_module) or a dict of strings and tuples (with specified (in_features, out_features)), got {config.target_modules}."
            )

    def create_lora_edges(self, model: nn.Module, config):
        """
        Create LoRA edges for the model.

        Args:
            model (nn.Module): The neural network model to control.
            edges (dict): 
        """
        if isinstance(config.target_modules, dict):
            edges = {k: v for k, v in config.target_modules.items() if isinstance(k, tuple) and len(k) == 2}
            nodes = {k: v for k, v in config.target_modules.items() if isinstance(k, str)}
        else:
            edges = [k for k in config.target_modules if isinstance(k, tuple) and len(k) == 2]
            nodes = [k for k in config.target_modules if isinstance(k, str)]
        if len(nodes)>0:
            origin_target_modules = config.target_modules
            config.target_modules = nodes
            dag_edges = super().create_lora_edges(model, config)
            config.target_modules = origin_target_modules
        else:
            dag_edges = {}
        submodules = dict(model.named_modules())
        if edges is not None:
            submodules = dict(model.named_modules())
            if isinstance(edges, dict):
                for (head, tail), (in_features, out_features) in edges.items():
                    heads = [k for k in submodules.keys() if k.endswith(head)]
                    tails = [k for k in submodules.keys() if k.endswith(tail)]
                    for h in heads:
                        for t in tails:
                            if h[:-len(head)] == t[:-len(tail)]:
                                dag_edges[(h, t)] = LoRAsideLayer(
                                    in_features=config.in_features,
                                    out_features=config.out_features,
                                    r=config.r,
                                    lora_alpha=config.lora_alpha,
                                    lora_dropout=config.lora_dropout,
                                    init_lora_weights=config.init_weights,
                                    use_rslora=config.use_rslora,
                                    lora_bias=config.lora_bias,
                                )
            elif isinstance(edges, list):
                for head, tail in edges:
                    heads = [k for k in submodules.keys() if k.endswith(head)]
                    tails = [k for k in submodules.keys() if k.endswith(tail)]
                    for h in heads:
                        for t in tails:
                            if h[:-len(head)] == t[:-len(tail)]:
                                in_features = submodules[h].in_features
                                out_features = submodules[t].out_features
                                dag_edges[(h, t)] = LoRAsideLayer(
                                    in_features=in_features,
                                    out_features=out_features,
                                    r=config.r,
                                    lora_alpha=config.lora_alpha,
                                    lora_dropout=config.lora_dropout,
                                    init_lora_weights=config.init_weights,
                                    use_rslora=config.use_rslora,
                                    lora_bias=config.lora_bias,
                                )
        return dag_edges




