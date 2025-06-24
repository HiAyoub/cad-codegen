import torch
import torch.nn as nn
from transformers import ViTModel

class EnhancedVisionEncoder(nn.Module):
    """Vision Transformer encoder with adaptive features"""
    
    def __init__(self, config):
        super().__init__()
        self.model = ViTModel.from_pretrained(
            config.model_name,
            add_pooling_layer=False
        )
        
        # Freeze first N layers if specified
        if config.get('frozen_layers', 0) > 0:
            for param in list(self.model.parameters())[:config.frozen_layers*12]:
                param.requires_grad = False
                
        self.projection = nn.Linear(768, 768)
        
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return self.projection(outputs.last_hidden_state)