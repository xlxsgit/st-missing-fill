#!/usr/bin/env python3
import sys
from src.utils import setup_logger
from src.data.processing import DataProcessor


def main():
    logger = setup_logger("main")

    try:
        logger.info("Starting data processing...")

        processor = DataProcessor()
        processor.process_and_save()
        processor.generate_all_stations()

        logger.info("Data processing completed successfully.")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
