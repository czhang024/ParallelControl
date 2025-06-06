import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaEncoder, RobertaModel, RobertaForSequenceClassification
from transformers import RobertaForSequenceClassification

class ControlledRobertaLayer(RobertaLayer):
    """Add control to the original Roberta layer"""
    def __init__(self, config, training_args):
        super().__init__(config)
        control_rank = training_args.control_rank  # since parallel method only controls 1 block instead of Q&V, it's more fair to set 2*r
        self.alpha = training_args.control_alpha
        self.device = "cuda"
        self.config = config
        self.seed = training_args.seed

        self.lora_A = nn.Linear(config.hidden_size, control_rank, bias=False).to(torch.float32)
        self.lora_B = nn.Linear(control_rank, config.hidden_size, bias=False).to(torch.float32)
            
    def reset_control_parameters(self):
        torch.manual_seed(self.seed)
        self.lora_A.weight = nn.Parameter(self.lora_A.weight.to(torch.float32))  # Ensure float32
        self.lora_B.weight = nn.Parameter(self.lora_B.weight.to(torch.float32))  # Ensure float32
        
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.zeros_(self.lora_B.weight)

        self.lora_A.weight.requires_grad = True
        self.lora_B.weight.requires_grad = True



    def feed_forward_chunk(self, attention_output):
        control_output = self.control_feed_forward(attention_output)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) + control_output
        return layer_output
    
    def control_feed_forward(self, x):
        down = self.lora_A(x)        
        up = self.lora_B(down)
        up = self.alpha * up
        return up


# from transformers.models.roberta.modeling_roberta import RobertaEncoder
class ControlledRobertaEncoder(RobertaEncoder):
    def __init__(self, config, training_args):
        super().__init__(config,)
        self.layer = nn.ModuleList(
            [ControlledRobertaLayer(config, training_args) for _ in range(config.num_hidden_layers)]
        )

class ControlledRobertaModel(RobertaModel):
    def __init__(self, config, training_args):
        super().__init__(config, add_pooling_layer=False)
        self.encoder = ControlledRobertaEncoder(config, training_args)

    def reset_controlled_layers(self):
        for layer in self.encoder.layer:
            if isinstance(layer, ControlledRobertaLayer):
                layer.reset_control_parameters()

class ControlledRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, training_args):
        super().__init__(config)
        self.roberta = ControlledRobertaModel(config, training_args)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.roberta.reset_controlled_layers()
        return model
