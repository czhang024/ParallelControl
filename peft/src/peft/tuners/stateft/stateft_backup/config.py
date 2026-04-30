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

from dataclasses import dataclass, field
from typing import Optional, Union, Literal

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class StateFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`StateFTModel`].

    Args:
        in_features (`int`):
            The input features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        out_features (`int`):
            The output features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        init_weights (`bool` ):
            Whether to perform initialization of adapter weights. This defaults to `True`. Use default initialization of pytorch. Passing `False` is discouraged.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        in_features_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `in_features`. For example, `{'^model.decoder.layers.0.mlp.down_proj': 4096}`.
        out_features_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `out_features`. For example, `{'^model.decoder.layers.0.mlp.up_proj': 4096}`.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. 
            For example, in Sequence Classification or Token Classification tasks, 
            the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved.
    """
    in_features: Optional[int] = field(
        default=None,
        metadata={
            "help": "The input features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually."
        }
    )
    out_features: Optional[int] = field(
        default=None,
        metadata={
            "help": "The output features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually."
        }
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with LoRA."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
            "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from LoRA."},
    )
    init_weights: Union[bool | Literal["gaussian"]] = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the LoRA layers with their default initialization. Can be True, False or 'lycoris'."
                "Default is True. Don't change this setting to False, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    in_features_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": "The mapping from layer names or regexp expression to ranks which are different from the default rank"
            "specified by `in_features`. For example, `{'^model.decoder.layers.0.mlp.down_proj': 4096}`."
        },
    )
    out_features_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": "The mapping from layer names or regexp expression to alphas which are different from the default alpha"
            "specified by `out_features`. For example, `{'^model.decoder.layers.0.mlp.up_proj': 4096}`."
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.STATEFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
         # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")


@dataclass
class StateFTLoraConfig(StateFTConfig):
    """
    This is the configuration class to store the configuration of a [`StateFTModel`].

    Args:
        in_features (`int`):
            The input features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        out_features (`int`):
            The output features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        r (`int`):
            LoRA rank.
        lora_alpha (`int`):
            The alpha parameter for LoRA scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        use_rslora (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `lora_alpha/r`.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        init_weights (`bool` ):
            Whether to perform initialization of adapter weights. This defaults to `True`. Use default initialization of pytorch. Passing `False` is discouraged.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        in_features_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `in_features`. For example, `{'^model.decoder.layers.0.mlp.down_proj': 4096}`.
        out_features_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `out_features`. For example, `{'^model.decoder.layers.0.mlp.up_proj': 4096}`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `alpha`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. 
            For example, in Sequence Classification or Token Classification tasks, 
            the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved.
        lora_bias (`bool`):
            Defaults to `False`. Whether to enable the bias term for the LoRA B parameter. Typically, this should be
            disabled. The main use case for this is when the LoRA weights were extracted from fully fine-tuned
            parameters so the bias of those parameters can be taken into account.
    """

    r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=8, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a>"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )
    lora_bias: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable the bias term for the LoRA B parameter. Typically, this should be disabled. The "
                "main use case for this is when the LoRA weights were extracted from fully fine-tuned parameters so "
                "the bias of those parameters can be taken into account."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.STATEFT_LORA
        if self.lora_bias:
            if self.init_weights not in (True, False):
                raise ValueError(
                    f"The argument lora_bias=True is only supported with init_weights=True or False, got "
                    f"init_weights={self.init_weights} instead."
                )


@dataclass
class StateFTv2Config(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`StateFTModel`].

    Args:
        in_features (`int`):
            The input features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        out_features (`int`):
            The output features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        init_weights (`bool` ):
            Whether to perform initialization of adapter weights. This defaults to `True`. Use default initialization of pytorch. Passing `False` is discouraged.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. 
            For example, in Sequence Classification or Token Classification tasks, 
            the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved.
    """
    in_features: Optional[int] = field(
        default=None,
        metadata={
            "help": "The input features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually."
        }
    )
    out_features: Optional[int] = field(
        default=None,
        metadata={
            "help": "The output features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually."
        }
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with LoRA."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
            "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
        },
    )
    init_weights: Union[bool | Literal["gaussian"]] = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the LoRA layers with their default initialization. Can be True, False or 'lycoris'."
                "Default is True. Don't change this setting to False, except if you know exactly what you're doing."
            ),
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.STATEFTV2
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        #  # if target_modules is a regex expression, then layers_to_transform should be None
        # if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
        #     raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # # if target_modules is a regex expression, then layers_pattern should be None
        # if isinstance(self.target_modules, str) and self.layers_pattern is not None:
        #     raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # # check for layers_to_transform and layers_pattern
        # if self.layers_pattern and not self.layers_to_transform:
        #     raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

@dataclass
class StateFTLorav2Config(StateFTv2Config):
    """
    This is the configuration class to store the configuration of a [`StateFTModel`].

    Args:
        in_features (`int`):
            The input features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        out_features (`int`):
            The output features of the module to apply the adapter to. Now it cannot be determined automatically, so it should be specified manually.
        r (`int`):
            LoRA rank.
        lora_alpha (`int`):
            The alpha parameter for LoRA scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        use_rslora (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `lora_alpha/r`.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        init_weights (`bool` ):
            Whether to perform initialization of adapter weights. This defaults to `True`. Use default initialization of pytorch. Passing `False` is discouraged.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. 
            For example, in Sequence Classification or Token Classification tasks, 
            the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved.
        lora_bias (`bool`):
            Defaults to `False`. Whether to enable the bias term for the LoRA B parameter. Typically, this should be
            disabled. The main use case for this is when the LoRA weights were extracted from fully fine-tuned
            parameters so the bias of those parameters can be taken into account.
    """

    r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=8, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a>"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    lora_bias: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable the bias term for the LoRA B parameter. Typically, this should be disabled. The "
                "main use case for this is when the LoRA weights were extracted from fully fine-tuned parameters so "
                "the bias of those parameters can be taken into account."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.STATEFT_LORA_V2
        if self.lora_bias:
            if self.init_weights not in (True, False):
                raise ValueError(
                    f"The argument lora_bias=True is only supported with init_weights=True or False, got "
                    f"init_weights={self.init_weights} instead."
                )