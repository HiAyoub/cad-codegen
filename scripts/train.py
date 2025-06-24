@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    setup_logging()
    
    # Detect hardware and adjust config
    has_gpu, hardware_config = detect_hardware()
    cfg = OmegaConf.merge(cfg, hardware_config)
    
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize WandB with hardware tags
    if cfg.use_wandb:
        wandb.init(
            project="cad-codegen",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["gpu" if has_gpu else "cpu"]
        )

    # Setup datasets with dynamic sample size
    train_dataset = CADDataset(
        split=cfg.data.train_samples,
        max_length=cfg.data.max_length
    )
    
    val_dataset = CADDataset(
        split=cfg.data.test_samples,
        max_length=cfg.data.max_length
    )
    
    # Adjust workers based on hardware
    num_workers = 4 if has_gpu else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=has_gpu
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        pin_memory=has_gpu
    )
    
    # Initialize model with appropriate device
    model = CADCodeGenerator().to(cfg.hardware.device)
    
    # Print model summary
    if has_gpu:
        from torchsummary import summary
        summary(model, [(3, 384, 384), (cfg.data.max_length,)], device=cfg.hardware.device)
    
    # Trainer with hardware-aware settings
    trainer = CADTrainer(model, cfg.training, cfg.hardware.device)
    
    # Training loop
    for epoch in range(cfg.training.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader)
        
        logger.info(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['eval_loss']:.4f} | "
            f"Device: {cfg.hardware.device.upper()}"
        )