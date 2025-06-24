import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, T5ForConditionalGeneration
from typing import Dict, Optional
from .components.vision_encoder import EnhancedVisionEncoder
from .components.code_decoder import SyntaxAwareDecoder


class CADCodeGenerator(nn.Module):
    """End-to-end CAD code generation model with syntax attention"""
    
    def __init__(self):
        super().__init__()
        self.vision_encoder = EnhancedVisionEncoder()
        self.code_decoder = SyntaxAwareDecoder()
        
        # Cross-modal projection
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        
        # Syntax attention gate
        self.syntax_gate = nn.Linear(768 * 2, 1)
        
    def forward(self, 
               pixel_values: torch.Tensor,
               input_ids: Optional[torch.Tensor] = None,
               attention_mask: Optional[torch.Tensor] = None,
               **kwargs) -> Dict[str, torch.Tensor]:
        
        # Encode image
        visual_features = self.vision_encoder(pixel_values)
        projected_visual = self.projection(visual_features)
        
        # Decode with syntax awareness
        outputs = self.code_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=projected_visual,
            **kwargs
        )
        
        # Apply syntax gating
        if input_ids is not None:
            syntax_weights = self._compute_syntax_weights(
                outputs.last_hidden_state,
                projected_visual
            )
            outputs.logits = outputs.logits * syntax_weights
        
        return outputs
    
    def _compute_syntax_weights(self, 
                              decoder_states: torch.Tensor,
                              visual_states: torch.Tensor) -> torch.Tensor:
        """Compute syntax attention weights"""
        visual_mean = visual_states.mean(dim=1, keepdim=True)
        visual_expanded = visual_mean.expand_as(decoder_states)
        
        combined = torch.cat([decoder_states, visual_expanded], dim=-1)
        return torch.sigmoid(self.syntax_gate(combined))