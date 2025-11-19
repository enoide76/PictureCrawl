"""
Logging configuration for Gemäldeagent.
"""
import sys
from pathlib import Path

from loguru import logger

from backend.core.config import settings


def setup_logging():
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # File handler
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if settings.LOG_FORMAT == "json":
        logger.add(
            settings.LOG_FILE,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            serialize=True,
        )
    else:
        logger.add(
            settings.LOG_FILE,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
        )

    logger.info("Logging configured successfully")
    return logger


# Initialize logging
log = setup_logging()
