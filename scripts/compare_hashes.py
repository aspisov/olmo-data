#!/usr/bin/env python3
"""
Compare SHA-256 hashes between two JSON files.

Usage:
    python scripts/compare_hashes.py processed_data/file_hashes.json reconstructed_data/file_hashes.json

Exits with 0 if hashes match, 2 otherwise.
"""

import json
import sys
from pathlib import Path


def load_hashes(path: str) -> dict[str, str]:
    """Load a {path: hash} mapping from *path*."""
    p = Path(path)
    if not p.exists():
        print(f"❌ File not found: {p}")
        sys.exit(2)
    with p.open() as f:
        return json.load(f)


def main() -> None:  # noqa: D401 – simple CLI
    if len(sys.argv) != 3:
        print("Usage: compare_hashes.py source_hashes.json target_hashes.json")
        sys.exit(1)

    src_path, tgt_path = sys.argv[1:3]
    src = load_hashes(src_path)
    tgt = load_hashes(tgt_path)

    missing = [k for k in src if k not in tgt]
    extra = [k for k in tgt if k not in src]
    mismatched = [k for k in src if k in tgt and src[k] != tgt[k]]

    if not missing and not extra and not mismatched:
        print("✅ All hashes match.")
        return

    if missing:
        print(f"❌ Missing in target ({len(missing)}):")
        for k in missing:
            print(f"   {k}")

    if extra:
        print(f"⚠️  Extra in target ({len(extra)}):")
        for k in extra:
            print(f"   {k}")

    if mismatched:
        print(f"❌ Hash mismatch ({len(mismatched)}):")
        for k in mismatched:
            print(f"   {k}")

    sys.exit(2)


if __name__ == "__main__":
    main()
