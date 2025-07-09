"""Logging utilities for the Knowledge Base System using loguru"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union

from loguru import logger


def setup_logger(
    name: Optional[str] = None,
    log_level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
) -> "logger":
    """
    Set up a logger with console and file handlers using loguru
    
    Args:
        name: Name of the logger context
        log_level: Logging level
        log_file: Path to the log file
        rotation: When to rotate the log file
        retention: How long to keep log files
        format: Log message format
        
    Returns:
        Configured logger
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=format,
        filter=lambda record: name is None or record["extra"].get("name", "") == name,
    )
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure the logs directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        logger.add(
            log_file,
            level=log_level,
            format=format,
            rotation=rotation,
            retention=retention,
            filter=lambda record: name is None or record["extra"].get("name", "") == name,
        )
    
    # Create a contextualized logger if name is provided
    if name:
        return logger.bind(name=name)
    
    return logger


def get_default_logger() -> "logger":
    """
    Get the default logger for the Knowledge Base System
    
    Returns:
        Default logger
    """
    # Get log file path from environment or use default
    log_file = os.getenv("LOG_FILE", "./logs/knowledge_base.log")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    return setup_logger("knowledge_base", log_level, log_file)


# Create a default logger
logger = get_default_logger() 