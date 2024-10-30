import sys
from datetime import datetime

import numpy as np
from loguru import logger


def get_logger(output_file: str = None):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger


def get_num_digits(number: int) -> int:
    return int(np.log10(number)) + 1


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
