"""Configuration utilities for the Knowledge Base System"""

import os
from typing import Dict, Any, Optional
import json
from pathlib import Path
from dotenv import load_dotenv

from knowledge_base.utils.logging import logger


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file and environment variables
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the merged configuration
    """
    # Load environment variables
    load_dotenv()
    
    # Default configuration
    config = {
        "api": {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
        },
        "database": {
            "mongodb_uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            "db_name": os.getenv("DB_NAME", "knowledge_base"),
        },
        "vector_db": {
            "path": os.getenv("VECTOR_DB_PATH", "./data/vector_db"),
            "use_cloud": os.getenv("USE_CHROMA_CLOUD", "false").lower() == "true",
            "cloud_url": os.getenv("CHROMA_CLOUD_URL", ""),
            "api_key": os.getenv("CHROMA_API_KEY", ""),
        },
        "extraction": {
            "max_tokens_per_doc": int(os.getenv("MAX_TOKENS_PER_DOC", "8000")),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        },
        "agent": {
            "model_name": os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "1024")),
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "file": os.getenv("LOG_FILE", "./logs/knowledge_base.log"),
        },
    }
    
    # Load configuration from file if provided
    if config_path:
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                config = deep_merge(config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure that required directories exist
    
    Args:
        config: Configuration dictionary
    """
    # Only create vector DB directory if not using cloud
    if not config.get("vector_db", {}).get("use_cloud", False):
        vector_db_path = config.get("vector_db", {}).get("path")
        if vector_db_path:
            Path(vector_db_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured vector database directory exists: {vector_db_path}")
    
    # Ensure data directory exists
    Path("./data").mkdir(exist_ok=True)
    logger.debug("Ensured data directory exists")
    
    # Ensure logs directory exists
    Path("./logs").mkdir(exist_ok=True)
    logger.debug("Ensured logs directory exists") 