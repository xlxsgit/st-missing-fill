#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to python path to allow importing src modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import load_config, setup_logger
from src.data.processing import DataProcessor

def main():
    logger = setup_logger('main')
    
    try:
        config_path = project_root / 'src' / 'config' / 'config.yaml'
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(str(config_path))
        
        processor = DataProcessor(config)
        processor.process_and_save()
        
        logger.info("Data processing completed successfully.")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
