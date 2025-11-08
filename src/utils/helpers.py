"""General helper utilities"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Union


def generate_image_id(image_path: Union[str, Path]) -> str:
    """
    Generate a unique ID for an image based on its content hash.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Unique image ID (16-character hex string)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    return file_hash[:16]


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0.0
    
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """
    Validate that a path points to a valid image file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        True if valid image, False otherwise
    """
    image_path = Path(image_path)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    return image_path.exists() and image_path.suffix in valid_extensions

