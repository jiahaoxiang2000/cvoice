"""File utility functions."""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_extension(file_path: str | Path) -> str:
    """Get file extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension including the dot
    """
    return Path(file_path).suffix.lower()


def validate_file_path(file_path: str | Path, must_exist: bool = True) -> Path:
    """Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid
    """
    path = Path(file_path)

    if must_exist:
        if not path.exists():
            raise ValueError(f"File does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

    return path


def get_unique_filename(base_path: str | Path, extension: str = "") -> Path:
    """Get unique filename by adding number suffix if needed.
    
    Args:
        base_path: Base file path
        extension: File extension to append
        
    Returns:
        Unique file path
    """
    base = Path(base_path)
    if extension and not extension.startswith('.'):
        extension = f".{extension}"

    # If no extension provided, use the original
    if not extension:
        full_path = base
    else:
        full_path = base.with_suffix(extension)

    if not full_path.exists():
        return full_path

    # Add number suffix until we find a unique name
    counter = 1
    while True:
        stem = base.stem
        suffix = base.suffix if not extension else extension
        new_name = f"{stem}_{counter}{suffix}"
        new_path = base.parent / new_name

        if not new_path.exists():
            return new_path

        counter += 1
