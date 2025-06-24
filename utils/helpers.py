import torch
from omegaconf import DictConfig
from typing import Tuple
from loguru import logger

def detect_hardware() -> Tuple[bool, DictConfig]:
    """Detect available hardware and return appropriate config"""
    has_gpu = torch.cuda.is_available()
    default_config = DictConfig({
        "hardware": {
            "use_gpu": has_gpu,
            "device": "cuda" if has_gpu else "cpu"
        }
    })
    
    if has_gpu:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return True, default_config
    else:
        logger.warning("No GPU detected - using CPU with reduced configuration")
        return False, _get_cpu_config(default_config)

def _get_cpu_config(base_config: DictConfig) -> DictConfig:
    """Create reduced configuration for CPU"""
    cpu_config = base_config.copy()
    
    # Reduce dataset size
    cpu_config.data.train_samples = "train[:10%]"
    cpu_config.data.test_samples = "test[:10%]"
    
    # Reduce model size
    cpu_config.model.vision_encoder.model_name = "google/vit-small-patch16-224"
    cpu_config.model.code_decoder.model_name = "Salesforce/codet5-small"
    
    # Reduce training parameters
    cpu_config.training.batch_size = 4
    cpu_config.training.mixed_precision = False
    cpu_config.training.gradient_accumulation_steps = 1
    
    return cpu_config