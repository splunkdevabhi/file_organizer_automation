"""
Utility functions for file operations in the file organizer.

This module provides helper functions for:
1. File type detection
2. Safe file operations (copy, move)
3. Path validation and normalization
4. File metadata extraction
"""

import os
import shutil
import logging
import mimetypes
import hashlib
import platform
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize mimetypes
mimetypes.init()


def get_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect the type of a file based on its extension and content.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File type category (image, document, video, etc.)
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get mimetype based on file extension
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    if mime_type is None:
        # If mimetype couldn't be determined from extension, try to determine from content
        if file_path.is_binary_file():
            return "binary"
        return "text"
    
    # Categorize the file based on its mimetype
    if mime_type.startswith("image/"):
        return "image"
    elif mime_type.startswith("video/"):
        return "video"
    elif mime_type.startswith("audio/"):
        return "audio"
    elif mime_type.startswith("text/"):
        return "document"
    elif mime_type == "application/pdf":
        return "document"
    elif "spreadsheet" in mime_type or "excel" in mime_type:
        return "spreadsheet"
    elif "presentation" in mime_type or "powerpoint" in mime_type:
        return "presentation"
    elif "archive" in mime_type or mime_type.endswith("zip") or mime_type.endswith("tar"):
        return "archive"
    elif mime_type.startswith("application/"):
        return "application"
    else:
        return "other"


def safe_copy_file(src_path: Union[str, Path], dest_path: Union[str, Path]) -> Path:
    """
    Safely copy a file, creating any necessary directories and handling name conflicts.
    
    Args:
        src_path: Source file path
        dest_path: Destination file path
        
    Returns:
        Path: Path to the copied file
        
    Raises:
        FileNotFoundError: If the source file does not exist
        PermissionError: If the file cannot be copied due to permissions
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    
    # Validate source file exists
    if not src_path.exists():
        logger.error(f"Source file not found: {src_path}")
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # Create destination directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle name conflicts
    if dest_path.exists():
        logger.info(f"File already exists at destination: {dest_path}")
        base_name = dest_path.stem
        extension = dest_path.suffix
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_dest_path = dest_path.parent / f"{base_name}_{timestamp}{extension}"
        logger.info(f"Renaming destination to: {new_dest_path}")
        dest_path = new_dest_path
    
    try:
        logger.info(f"Copying {src_path} to {dest_path}")
        shutil.copy2(src_path, dest_path)
        return dest_path
    except PermissionError as e:
        logger.error(f"Permission error when copying {src_path} to {dest_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error copying {src_path} to {dest_path}: {e}")
        raise


def safe_move_file(src_path: Union[str, Path], dest_path: Union[str, Path]) -> Path:
    """
    Safely move a file, creating any necessary directories and handling name conflicts.
    
    Args:
        src_path: Source file path
        dest_path: Destination file path
        
    Returns:
        Path: Path to the moved file
        
    Raises:
        FileNotFoundError: If the source file does not exist
        PermissionError: If the file cannot be moved due to permissions
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    
    # Validate source file exists
    if not src_path.exists():
        logger.error(f"Source file not found: {src_path}")
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # Create destination directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle name conflicts
    if dest_path.exists():
        logger.info(f"File already exists at destination: {dest_path}")
        base_name = dest_path.stem
        extension = dest_path.suffix
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_dest_path = dest_path.parent / f"{base_name}_{timestamp}{extension}"
        logger.info(f"Renaming destination to: {new_dest_path}")
        dest_path = new_dest_path
    
    try:
        logger.info(f"Moving {src_path} to {dest_path}")
        shutil.move(src_path, dest_path)
        return dest_path
    except PermissionError as e:
        logger.error(f"Permission error when moving {src_path} to {dest_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error moving {src_path} to {dest_path}: {e}")
        raise


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a file path to an absolute path with resolved symlinks.
    
    Args:
        path: Path to normalize
        
    Returns:
        Path: Normalized path
    """
    path = Path(path).expanduser().absolute().resolve()
    return path


def validate_path(path: Union[str, Path], must_exist: bool = True, 
                 must_be_file: bool = False, must_be_dir: bool = False) -> Path:
    """
    Validate a file path based on various criteria.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        must_be_file: Whether the path must be a file
        must_be_dir: Whether the path must be a directory
        
    Returns:
        Path: Validated path
        
    Raises:
        FileNotFoundError: If the path does not exist and must_exist is True
        NotADirectoryError: If the path is not a directory and must_be_dir is True
        IsADirectoryError: If the path is a directory and must_be_file is True
    """
    path = normalize_path(path)
    
    if must_exist and not path.exists():
        logger.error(f"Path does not exist: {path}")
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if must_be_dir and not path.is_dir():
        logger.error(f"Path is not a directory: {path}")
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    if must_be_file and not path.is_file():
        logger.error(f"Path is not a file: {path}")
        raise IsADirectoryError(f"Path is not a file: {path}")
    
    return path


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict[str, Any]: Dictionary containing file metadata
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Get basic file stats
        stat_info = file_path.stat()
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        
        metadata = {
            "name": file_path.name,
            "path": str(file_path.absolute()),
            "size": stat_info.st_size,
            "size_human": format_file_size(stat_info.st_size),
            "created": datetime.fromtimestamp(stat_info.st_ctime),
            "modified": datetime.fromtimestamp(stat_info.st_mtime),
            "accessed": datetime.fromtimestamp(stat_info.st_atime),
            "extension": file_path.suffix.lower(),
            "type": get_file_type(file_path),
            "hash": file_hash,
            "is_hidden": is_hidden_file(file_path)
        }
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
        raise


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256', 
                      buffer_size: int = 65536) -> str:
    """
    Calculate the hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        buffer_size: Size of the buffer for reading the file
        
    Returns:
        str: Hexadecimal hash string
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            buffer = f.read(buffer_size)
            while buffer:
                hash_obj.update(buffer)
                buffer = f.read(buffer_size)
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        raise


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        str: Human-readable file size
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} {unit}"


def is_hidden_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is hidden.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is hidden, False otherwise
    """
    file_path = Path(file_path)
    
    # On Unix/Linux/Mac, hidden files start with a dot
    if platform.system() != "Windows":
        return file_path.name.startswith('.')
    
    # On Windows, use file attributes
    import ctypes
    
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(file_path))
        assert attrs != -1
        return bool(attrs & 2)
    except (AttributeError, AssertionError):
        return False


def find_duplicates(directory: Union[str, Path], recursive: bool = True) -> Dict[str, List[Path]]:
    """
    Find duplicate files in a directory based on their content hash.
    
    Args:
        directory: Directory to search for duplicates
        recursive: Whether to search recursively
        
    Returns:
        Dict[str, List[Path]]: Dictionary mapping file hashes to lists of file paths
        
    Raises:
        NotADirectoryError: If the specified path is not a directory
    """
    directory = validate_path(directory, must_exist=True, must_be_dir=True)
    
    # Dictionary to store file hashes and their paths
    file_hashes: Dict[str, List[Path]] = {}
    
    logger.info(f"Searching for duplicates in {directory}")
    
    # Walk through the directory
    walk_pattern = directory.rglob('*') if recursive else directory.glob('*')
    
    for path in walk_pattern:
        if path.is_file():
            try:
                file_hash = calculate_file_hash(path)
                
                if file_hash in file_hashes:
                    file_hashes[file_hash].append(path)
                else:
                    file_hashes[file_hash] = [path]
            except Exception as e:
                logger.warning(f"Error processing file {path}: {e}")
    
    # Filter out non-duplicates
    duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
    
    logger.info(f"Found {sum(len(p) for p in duplicates.values()) - len(duplicates)} duplicate files")
    
    return duplicates


def wait_for_file_access(file_path: Union[str, Path], timeout: int = 10) -> bool:
    """
    Wait until a file is accessible (not being written to).
    
    Args:
        file_path: Path to the file
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if the file became accessible, False if timeout was reached
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    start_time = time.time()
    prev_size = -1
    
    while time.time() - start_time < timeout:
        try:
            current_size = file_path.stat().st_size
            if current_size == prev_size and prev_size != -1:
                # File size hasn't changed, likely not being written to
                return True
            
            prev_size = current_size
            time.sleep(0.5)
        except Exception:
            # If we can't access the file, it might be locked
            time.sleep(0.5)
    
    logger.warning(f"Timeout reached waiting for file access: {file_path}")
    return False

