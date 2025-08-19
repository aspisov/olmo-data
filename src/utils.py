#!/usr/bin/env python3
"""
Shared utilities for data processing and reconstruction.

This module centralizes helpers that are used by multiple scripts to keep the
core logic files shorter and easier to maintain.
"""

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path (Path): Path to the file on disk.

    Returns:
        str: Hex-encoded SHA256 hash.
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def load_binary_data(file_path: Path) -> np.ndarray:
    """
    Load binary data assuming .npy format; fallback to raw uint32 buffer.

    Args:
        file_path (Path): Path to the input file.

    Returns:
        np.ndarray: Memory-mapped array when possible.
    """
    try:
        return np.memmap(file_path, mode="r", dtype=np.uint32)
    except Exception:
        raise ValueError(f"Failed to load binary data from {file_path}")


def extract_part_index(filename: str) -> str:
    """
    Extract part index from filename, preferring the full filename pattern.

    Args:
        filename (str): File name like "part-00-00000.npy".

    Returns:
        str: Part index string (usually the full file name).
    """
    name = Path(filename).name

    if name.startswith("part-") and name.endswith(".npy"):
        # Remove "part-" prefix and ".npy" suffix → "<index>" (e.g. "4-00000")
        return name[len("part-") : -len(".npy")]
    raise ValueError(f"Invalid filename: {filename}")


def extract_dataset_path(url: str) -> str:
    """
    Extract the dataset path between 'preprocessed' and 'part-' from a URL.

    Args:
        url (str): Source URL.

    Returns:
        str: Relative dataset path ending with a trailing slash.
    """
    try:
        pre_idx = url.find("/preprocessed/")
        if pre_idx == -1:
            return Path(urlparse(url).path).name.replace(".npy", "")

        tail = url[pre_idx + len("/preprocessed/") :]
        part_idx = tail.rfind("/part-")
        if part_idx == -1:
            return tail.rstrip("/")

        dataset_path = tail[: part_idx + 1]
        return dataset_path
    except Exception:
        return Path(urlparse(url).path).name.replace(".npy", "")


def encoded_csv_name_from_group(group_name: str) -> str:
    """
    Encode group path into a filesystem-safe CSV file name.

    Args:
        group_name (str): Group path like "a/b/c/".

    Returns:
        str: Encoded base name like "a__b__c".
    """
    return group_name.replace("/", "__").replace(":", "").rstrip("_")
