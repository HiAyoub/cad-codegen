import logging
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
from loguru import logger
import sys
from typing import Optional
from pathlib import Path
import json
import torch

class CADLogger:
    """Unified logging system for training and evaluation"""
    
    def __init__(self, config: dict, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure Loguru
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=config.get("log_level", "INFO")
        )
        
        # File logging
        logger.add(
            self.log_dir / "training.log",
            rotation="10 MB",
            retention="7 days",
            level="DEBUG"
        )
        
        # Hardware info
        self._log_hardware()
    
    def _log_hardware(self):
        """Log hardware configuration"""
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
    
    @staticmethod
    def log_config(config: dict):
        """Pretty print configuration"""
        logger.info("Running with configuration:")
        logger.info(json.dumps(config, indent=4, sort_keys=True))
    
    @staticmethod
    def log_metrics(epoch: int, metrics: dict, prefix: str = "train"):
        """Format and log training metrics"""
        log_str = f"Epoch {epoch} | "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"[{prefix.upper()}] {log_str}")
    
    @staticmethod
    def log_exception(error: Exception, context: Optional[str] = None):
        """Standardized error logging"""
        logger.opt(exception=True).error(f"{context + ' ' if context else ''}Error: {error}")

# Singleton pattern for easy access
logging_config = {
    "log_level": "INFO",
    "log_format": "simple"
}

logger = CADLogger(logging_config)