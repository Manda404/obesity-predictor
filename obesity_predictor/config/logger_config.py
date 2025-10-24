"""
logger_config.py
=========================
Centralized logging configuration using Loguru.

Features
--------
- Colorized console logs
- File logs with rotation, retention, and compression
- Uniform format across all modules
- Easy integration with production (FastAPI, ML pipelines, etc.)

Author: Rostand Surel
"""

from loguru import logger
from pathlib import Path
import sys
from obesity_predictor.config.settings import settings


# Create logs directory if it doesn't exist
LOG_DIR = Path(settings.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove default handlers (avoid duplicate logs)
logger.remove()

# ---- Console handler (color + clean output)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO",
)

# ---- File handler (rotating log file)
logger.add(
    LOG_DIR / "obesity_predictor.log",
    rotation="10 MB",            # rotate every 10 MB
    retention="10 days",         # keep logs for 10 days
    compression="zip",           # compress old logs
    enqueue=True,                # safe for multiprocessing
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# Export the configured logger
__all__ = ["logger"]
