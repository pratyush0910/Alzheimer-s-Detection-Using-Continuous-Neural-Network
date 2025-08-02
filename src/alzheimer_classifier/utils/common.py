import os
from box.exceptions import BoxValueError
import yaml
from alzheimer_classifier import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns its content as a ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """Get the size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: The size of the file in KB as a string.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"