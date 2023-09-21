import logging
import sys

from loguru import logger

logging.getLogger("xformers").setLevel(logging.ERROR)


def _patcher(record):
    record["extra"]["context"] = ", ".join(
        (f"{k}={v}" for k, v in record["extra"].items())
    )
    return record


def setup_logging():
    return logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message} ({extra[context]})</level>",
                "colorize": True,
            }
        ],
        patcher=_patcher,
    )
