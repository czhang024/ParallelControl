import torch
import torch.nn as nn
import math
import os
from typing import Optional, Tuple, Dict
from transformers import LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from ..utils import get_peft_model_state_dict
from ..utils import PeftConfig, PeftType
from dataclasses import dataclass, field


@dataclass
class ControlConfig(PeftConfig):
    control_rank: int = field(default=64, metadata={"help": "Lora attention dimension"})
    control_alpha: float = field(default=1.0, metadata={"help": "Control alpha"})
    double_control: bool = field(default=False, metadata={"help": "Whether use double control."})
    double_control_rank: int = field(default=64, metadata={"help": "Attention Control Rank"})
    control_dropout: float = field(default=0.0, metadata={"help": "Control dropout rate"})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: list = field(default_factory=list, metadata={"help": "Modules to save"})
    

    def __post_init__(self):
        self.peft_type = PeftType.CONTROL
        self.modules_to_save = []


class ControlledLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, control_args):
        super().__init__(config, layer_idx)
        self.control_rank = control_args["control_rank"]
        self.double_control = control_args["double_control"]
        self.control_alpha =  control_args["control_alpha"]
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.lora_A = nn.Linear(self.hidden_size, self.control_rank, bias=False).to(dtype=torch.float32)
        self.lora_B = nn.Linear(self.control_rank, self.hidden_size, bias=False).to(dtype=torch.float32)
        self.lora_dropout = nn.Dropout(p=control_args["control_dropout"])
        # self.lora_dropout = nn.Dropout(p=0.05)
        if self.double_control: # whether also use control on the attn block
            self.lora_A_attn = nn.Linear(self.hidden_size,control_args["double_control_rank"],bias=False).to(dtype=torch.float32)
            self.lora_B_attn = nn.Linear(control_args["double_control_rank"],self.hidden_size,bias=False).to(dtype=torch.float32)
            self.lora_attn_dropout = nn.Dropout(p=control_args["control_dropout"])
        

    def reset_control_parameters(self):
        """
        Resets the parameters of LoRA layers (lora_A and lora_B).
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))  # Kaiming uniform initialization
        nn.init.zeros_(self.lora_B.weight)  # Reset lora_B weight to zero
        if self.double_control:
            nn.init.kaiming_uniform_(self.lora_A_attn.weight, a=math.sqrt(5))  # Kaiming uniform initialization
            nn.init.zeros_(self.lora_B_attn.weight)  # Reset lora_B weight to zero

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        if self.double_control:
            control_attn = self.lora_B_attn(self.lora_attn_dropout(self.lora_A_attn(hidden_states)))
        hidden_states = self.input_layernorm(hidden_states)
        
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states 
        if self.double_control:
            hidden_states += self.control_alpha * control_attn

        # Fully Connected (with control)
        residual = hidden_states
        control = self.lora_B(self.lora_dropout(self.lora_A(hidden_states))) 
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states + self.control_alpha * control

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class ControlledLlamaModel(LlamaModel):
    def __init__(self, config, control_args):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ControlledLlamaDecoderLayer(config, layer_idx, control_args) for layer_idx in range(config.num_hidden_layers)]
        )
    
    def reset_control_layers(self):
        """
        Reset LoRA parameters for all decoder layers.
        """
        for layer in self.layers:
            if isinstance(layer, ControlledLlamaDecoderLayer):
                layer.reset_control_parameters()

class ControlledLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, control_args):
        super().__init__(config)
        self.model = ControlledLlamaModel(config, control_args)
        self.peft_config = ControlConfig(**control_args)
        self.modules_to_save = self.peft_config.modules_to_save

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.model.reset_control_layers()
        return model
    
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, "model.bin"))

        # save the config and change the inference mode to `True`
        if self.peft_config.base_model_name_or_path is None:
            self.peft_config.base_model_name_or_path = (
                self.model.__dict__.get("name_or_path", None)
            )
        inference_mode = self.peft_config.inference_mode
        self.peft_config.inference_mode = True
        self.peft_config.save_pretrained(save_directory)
        self.peft_config.inference_mode = inference_mode