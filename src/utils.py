import yaml
import logging
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.absolute()}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger(name: str = 'processing', level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with standard formatting.

    Args:
        name (str): Name of the logger.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
