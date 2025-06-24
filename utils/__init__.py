from .log_utils import CADLogger, logger
__all__ = []
from .hardware import detect_hardware, configure_for_hardware
from .log_utils import setup_logging

__all__ = [
    'detect_hardware',
    'configure_for_hardware',
    'setup_logging',
    'CADLogger', 
    'logger'
]