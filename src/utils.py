"""
Utility functions for the house price prediction project.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def ensure_dir_exists(path: Path) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def save_data(data: pd.DataFrame, filepath: Path) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        data: DataFrame to save
        filepath: Path to save file
    """
    ensure_dir_exists(filepath.parent)
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        Loaded DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = pd.read_csv(filepath)
    print(f"Data loaded from {filepath} - Shape: {data.shape}")
    return data


def print_data_info(data: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic information about a DataFrame.

    Args:
        data: DataFrame to analyze
        name: Name for the dataset
    """
    print(f"\n{name} Information:")
    print(f"Shape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
