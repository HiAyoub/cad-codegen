import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from loguru import logger
from tqdm import tqdm
from typing import Dict, List
import numpy as np
import wandb


class CADTrainer:
    def __init__(self, model, training_config, device):
        self.model = model
        self.config = training_config
        self.device = device
        
        # Hardware-aware optimization
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision and "cuda" in self.device)
        
        # Adjust optimizer based on hardware
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Fewer warmup steps on CPU
        warmup_steps = 100 if "cuda" in self.device else 20
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.epochs * len(train_loader)
        )

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            
            # Mixed precision only on GPU
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            if "cuda" in self.device:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            total_loss += loss.item()
            
        return {"train_loss": total_loss / len(dataloader)}
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                # Calculate additional metrics
                metrics = self._calculate_metrics(outputs, batch)
                all_metrics.append(metrics)
        
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        avg_metrics["eval_loss"] = total_loss / len(dataloader)
        
        return avg_metrics
    
    def _calculate_metrics(self, outputs, batch) -> Dict[str, float]:
        """Calculate syntax accuracy and other custom metrics"""
        # Implementation depends on your specific metrics
        return {"syntax_acc": 0.0}  # Placeholder