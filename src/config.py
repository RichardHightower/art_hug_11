"""Configuration module for examples."""

import os
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
import warnings

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configurations
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bert-base-uncased")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# API keys (if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def validate_api_key(key_name: str, key_value: Optional[str], required: bool = False) -> bool:
    """
    Validate an API key.
    
    Args:
        key_name: Name of the API key (for error messages)
        key_value: The actual API key value
        required: Whether the key is required for operation
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        ValueError: If required key is missing or invalid
    """
    if not key_value:
        if required:
            raise ValueError(f"{key_name} is required but not set in environment variables")
        else:
            warnings.warn(f"{key_name} not found in environment variables", UserWarning)
            return False
    
    # Basic validation - check if it's not just whitespace
    if not key_value.strip():
        if required:
            raise ValueError(f"{key_name} is set but empty")
        return False
    
    # Check for placeholder values
    if key_value.lower() in ["your-api-key-here", "placeholder", "xxx", "todo"]:
        if required:
            raise ValueError(f"{key_name} contains a placeholder value")
        warnings.warn(f"{key_name} contains a placeholder value", UserWarning)
        return False
    
    return True


# Validate API keys on import (optional validation)
validate_api_key("OPENAI_API_KEY", OPENAI_API_KEY, required=False)
validate_api_key("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY, required=False)
validate_api_key("HUGGINGFACE_TOKEN", HF_TOKEN, required=False)


# Device configuration
import torch


def get_device() -> Literal["mps", "cuda", "cpu"]:
    """
    Get the best available device for PyTorch computation.
    
    Returns:
        str: Device string - "mps" for Apple Silicon, "cuda" for NVIDIA GPU, or "cpu"
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
        

DEVICE = get_device()
