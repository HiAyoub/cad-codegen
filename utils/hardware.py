import torch
from omegaconf import DictConfig
from loguru import logger
from typing import Tuple

def detect_hardware() -> Tuple[bool, DictConfig]:
    """Detect available hardware and return appropriate config"""
    has_gpu = torch.cuda.is_available()
    
    hardware_config = DictConfig({
        "hardware": {
            "use_gpu": has_gpu,
            "device": "cuda" if has_gpu else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if has_gpu else None,
            "cuda_version": torch.version.cuda if has_gpu else None
        }
    })
    
    if has_gpu:
        logger.info(f"GPU detected: {hardware_config.hardware.gpu_name}")
    else:
        logger.warning("No GPU detected - using CPU configuration")
    
    return has_gpu, hardware_config

def configure_for_hardware(cfg: DictConfig) -> DictConfig:
    """Adjust configuration based on hardware"""
    has_gpu, hardware_config = detect_hardware()
    cfg = OmegaConf.merge(cfg, hardware_config)
    
    if not has_gpu:
        # Reduce parameters for CPU
        cfg.data.train_samples = "train[:10%]"
        cfg.data.test_samples = "test[:10%]"
        cfg.training.batch_size = min(cfg.training.batch_size, 4)
        cfg.training.mixed_precision = False
        
    return cfg