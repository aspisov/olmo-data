from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import yaml


def _extract_group_key(url: str) -> str:
    """
    Extract the grouping key from a URL.

    The key is the directory path after "/preprocessed/" up to and including the
    directory that contains the part files, e.g.,
    "dclm/v0_rep32_ft7percentile_fw2/documents/allenai/dolma2-tokenizer/0000/".

    Args:
        url (str): Full URL string.

    Returns:
        str: Group key with a trailing slash.
    """
    parsed = urlparse(url)
    path = parsed.path
    if not path:
        return ""

    # Find the segment after "/preprocessed/" if present
    marker = "/preprocessed/"
    try:
        after = path.split(marker, 1)[1]
    except IndexError:
        after = path.lstrip("/")

    # Drop filename and ensure trailing slash
    if "/" in after:
        directory = after.rsplit("/", 1)[0]
    else:
        directory = after
    if not directory.endswith("/"):
        directory += "/"
    return directory


def group_paths(urls: List[str]) -> Dict[str, List[str]]:
    """
    Group URLs by their directory after "/preprocessed/" and sort groups and items.

    Args:
        urls (List[str]): List of URL strings.

    Returns:
        Dict[str, List[str]]: Ordered mapping of group key -> sorted list of URLs.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for url in urls:
        key = _extract_group_key(url)
        if not key:
            continue
        groups[key].append(url)

    # Sort items within each group lexicographically
    for key, items in groups.items():
        items.sort()

    # Build a plain dict with sorted keys to preserve order while remaining YAML-serializable
    return {key: groups[key] for key in sorted(groups.keys())}


def load_urls_from_yaml(path: Path) -> List[str]:
    """
    Load the list of URLs from a YAML file under the top-level key "paths".

    Args:
        path (Path): YAML file path.

    Returns:
        List[str]: List of URL strings.
    """
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    urls = data.get("paths", [])
    if not isinstance(urls, list):
        raise ValueError("Expected 'paths' to be a list in the YAML input")
    # Keep only strings
    return [u for u in urls if isinstance(u, str)]


def dump_grouped_to_yaml(grouped: Dict[str, List[str]]) -> str:
    """
    Dump grouped mapping under top-level key "paths" as YAML.

    Args:
        grouped (Dict[str, List[str]]): Ordered mapping from group -> list of URLs.

    Returns:
        str: YAML string.
    """
    payload = {"paths": grouped}
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def main() -> None:
    """
    CLI entrypoint to group paths from a YAML file and output grouped YAML.
    """
    parser = argparse.ArgumentParser(
        description="Group OLMO dataset paths by directory"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("all_paths.yaml"),
        help="Input YAML file with a top-level 'paths' list (default: all_paths.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output YAML file. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    urls = load_urls_from_yaml(args.input)
    grouped = group_paths(urls)
    yaml_text = dump_grouped_to_yaml(grouped)

    if args.output:
        args.output.write_text(yaml_text)
    else:
        print(yaml_text, end="")


if __name__ == "__main__":
    main()
