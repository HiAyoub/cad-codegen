import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from loguru import logger
import wandb

from models.syntax_aware import CADCodeGenerator
from data.dataset import CADDataset
from training.trainer import CADTrainer
from utils.log_utils import setup_logging
from utils.hardware import detect_hardware, configure_for_hardware
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader

import sys
print("PYTHONPATH:", sys.path)

@hydra.main(version_base=None,config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    setup_logging()
    logger.info("Logging setup complete.")
    
    # Detect hardware and adjust config
    logger.info("Detecting hardware...")
    has_gpu, hardware_config = detect_hardware()
    cfg = OmegaConf.merge(cfg, hardware_config)
    logger.info(f"Hardware detected: {'GPU' if has_gpu else 'CPU'}")
    
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize WandB with hardware tags
    if cfg.use_wandb:
        logger.info("Initializing WandB...")
        wandb.init(
            project="cad-codegen",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["gpu" if has_gpu else "cpu"]
        )
        logger.info("WandB initialized.")

    # Setup datasets with dynamic sample size
    logger.info("Setting up datasets...")
    train_dataset = CADDataset(
        split=cfg.data.train_samples,
        max_length=cfg.data.max_length
    )
    logger.info("Training dataset initialized.")
    
    val_dataset = CADDataset(
        split=cfg.data.test_samples,
        max_length=cfg.data.max_length
    )
    logger.info("Validation dataset initialized.")
    
    # Adjust workers based on hardware
    num_workers = 4 if has_gpu else 2
    logger.info(f"Number of workers set to {num_workers}.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=has_gpu
    )
    logger.info("Training DataLoader initialized.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        pin_memory=has_gpu
    )
    logger.info("Validation DataLoader initialized.")
    
    # Initialize model with appropriate device
    logger.info("Initializing model...")
    model = CADCodeGenerator().to(cfg.hardware.device)
    logger.info("Model initialized and moved to device.")
    
    # Print model summary
    if has_gpu:
        logger.info("Generating model summary...")
        from torchsummary import summary
        summary(model, [(3, 384, 384), (cfg.data.max_length,)], device=cfg.hardware.device)
        logger.info("Model summary generated.")
    
    # Trainer with hardware-aware settings
    logger.info("Initializing trainer...")
    trainer = CADTrainer(model, cfg.training, cfg.hardware.device)
    logger.info("Trainer initialized.")
    
    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(cfg.training.epochs):
        logger.info(f"Starting epoch {epoch+1}/{cfg.training.epochs}...")
        train_metrics = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Epoch {epoch+1} training completed. Train Loss: {train_metrics['train_loss']:.4f}")
        
        val_metrics = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch+1} validation completed. Val Loss: {val_metrics['eval_loss']:.4f}")
        
        logger.info(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['eval_loss']:.4f} | "
            f"Device: {cfg.hardware.device.upper()}"
        )
    logger.info("Training loop completed.")

if __name__ == "__main__":
    main()
