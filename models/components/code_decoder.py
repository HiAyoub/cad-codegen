import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class SyntaxAwareDecoder(nn.Module):
    """Code generator with syntax attention"""
    
    def __init__(self, config):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.model_name
        )
        
        # Add special tokens if needed
        if config.get('add_special_tokens', True):
            self.model.resize_token_embeddings(config.vocab_size)
            
    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )