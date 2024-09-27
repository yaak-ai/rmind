import logging  # noqa: A005
import sys

from loguru import logger


def setup_logging() -> None:
    logging.getLogger("xformers").setLevel(logging.ERROR)

    _ = logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message} | {extra}</level>",
                "colorize": True,
            }
        ]
    )
