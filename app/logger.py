from loguru import logger
import sys


def setup_logger():
    logger.remove()  # Remove default logger

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
