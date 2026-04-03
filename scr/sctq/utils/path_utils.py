from pathlib import Path
from typing import Union


def get_base_name(path: Union[str, Path]) -> str:
    """Get the base name of a file or directory without extension."""
    return Path(path).stem


def get_file_extension(path: Union[str, Path]) -> str:
    """Get the extension of a file."""
    return Path(path).suffix
