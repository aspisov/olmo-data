#!/usr/bin/env python3
"""
Data Reconstruction Script

Converts processed CSV files with part columns back to .npy format and recreates
the original directory structure.

Usage:
    python src/data_reconstructor.py processed_data/
    python src/data_reconstructor.py --input-dir processed_data/ --output-dir reconstructed_data/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl
from tqdm import tqdm

# Local utilities
from utils import calculate_file_hash


class DataReconstructor:
    """Reconstructs original dataset structure from processed CSV files."""

    def __init__(
        self,
        input_dir: str = "processed_data",
        output_dir: str = "reconstructed_data",
        skip_existing: bool = True,
    ):
        """
        Initialize the data reconstructor.

        Args:
            input_dir (str): Directory containing processed CSV files.
            output_dir (str): Directory to recreate original structure.
            skip_existing (bool): Whether to skip already reconstructed files.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.skip_existing = skip_existing

        # Create directories
        self.output_dir.mkdir(exist_ok=True)

        # Track reconstructed files and their hashes
        self.reconstructed_hashes: Dict[str, str] = {}
        self.file_count = 0

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        return calculate_file_hash(file_path)

    def group_csv_files(self, csv_files: List[Path]) -> Dict[str, List[Path]]:
        """
        Group CSV files by their base name (handling chunks).

        Args:
            csv_files (List[Path]): List of CSV file paths.

        Returns:
            Dict[str, List[Path]]: Grouped files by base name.
        """
        groups = {}

        for csv_path in csv_files:
            # Extract base name (remove _chunk_X suffix if present)
            name = csv_path.stem
            if "_chunk_" in name:
                base_name = name.split("_chunk_")[0]
            else:
                base_name = name

            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(csv_path)

        # Sort files within each group to ensure proper order
        for base_name in groups:
            groups[base_name].sort(
                key=lambda p: (
                    0 if "_chunk_" not in p.stem else int(p.stem.split("_chunk_")[1])
                )
            )

        return groups

    def extract_group_path_from_filename(self, csv_filename: str) -> str:
        """
        Extract the original group path from CSV filename by decoding __ back to /.

        Args:
            csv_filename (str): Name of CSV file (e.g., "basic_math_mj__multiadd__dolma2-tokenizer.csv")

        Returns:
            str: Group path (e.g., "basic_math_mj/multiadd/dolma2-tokenizer/")
        """
        base_name = csv_filename.replace(".csv", "")
        decoded_path = base_name.replace("__", "/")

        if not decoded_path.endswith("/"):
            decoded_path += "/"

        return decoded_path

    def reconstruct_filename_from_index(self, part_index: str) -> str:
        """
        Reconstruct the original filename from part index.

        Args:
            part_index (str): Part index. If already a filename like "part-*.npy",
                it is returned as-is. Otherwise, fall back to legacy pattern.

        Returns:
            str: Filename for the part.
        """
        # If the index already looks like a full filename keep it untouched
        if part_index.startswith("part-") and part_index.endswith(".npy"):
            return part_index

        # Compact index case (e.g. "4-00000") → restore original pattern
        return f"part-{part_index}.npy"

    def reconstruct_npy_from_part(
        self, part_df: pl.DataFrame, part_index: str
    ) -> np.ndarray | None:
        """
        Reconstruct .npy data from a part DataFrame.

        Args:
            part_df (pl.DataFrame): DataFrame containing data for one part.
            part_index (str): Index of the part.

        Returns:
            np.ndarray | None: Reconstructed numpy array, or None if failed.
        """
        try:
            columns = part_df.columns

            if len(columns) == 2 and "value" in columns and "part" in columns:
                # Standard 1D array format – preserve original dtype
                data = part_df["value"].to_numpy()

            elif "row_idx" in columns and "col_idx" in columns:
                # 2D array format (not currently used but kept for future)
                values = part_df["value"].to_numpy()

                if "original_shape" in columns:
                    shape_str = part_df["original_shape"][0]
                    shape = eval(shape_str)
                    data = np.zeros(shape, dtype=np.uint32)

                    row_indices = part_df["row_idx"].to_numpy()
                    col_indices = part_df["col_idx"].to_numpy()

                    for i, (row_idx, col_idx, value) in enumerate(
                        zip(row_indices, col_indices, values)
                    ):
                        data[row_idx, col_idx] = value
                else:
                    max_row_val = part_df["row_idx"].max()
                    max_col_val = part_df["col_idx"].max()
                    if max_row_val is None or max_col_val is None:
                        print(f"  ❌ Cannot determine shape for part {part_index}")
                        return None

                    max_row = int(part_df["row_idx"].to_numpy().max()) + 1
                    max_col = int(part_df["col_idx"].to_numpy().max()) + 1
                    data = np.zeros((max_row, max_col), dtype=np.uint32)

                    row_indices = part_df["row_idx"].to_numpy()
                    col_indices = part_df["col_idx"].to_numpy()

                    for i, (row_idx, col_idx, value) in enumerate(
                        zip(row_indices, col_indices, values)
                    ):
                        data[row_idx, col_idx] = values[i]

            elif "original_shape" in columns:
                # Higher dimensions (not currently used but kept for future)
                values = part_df["value"].to_numpy()
                shape_str = part_df["original_shape"][0]
                shape = eval(shape_str)
                data = values.reshape(shape)

            elif "value" in columns:
                # Fallback: use value column
                data = part_df["value"].to_numpy()
            else:
                print(f"  ❌ Unknown format for part {part_index}: {columns}")
                return None

            # Ensure dtype matches source (OLMo datasets are uint32)
            if data.dtype != np.uint32:
                data = data.astype(np.uint32)

            print(
                f"  ✅ Reconstructed part {part_index}: {data.shape}, dtype={data.dtype}"
            )
            return data

        except Exception as e:
            print(f"  ❌ Failed to reconstruct part {part_index}: {e}")
            return None

    def is_file_already_reconstructed(self, output_path: Path) -> bool:
        """Check if a file has already been reconstructed."""
        return output_path.exists()

    def process_csv_file(self, csv_path: Path) -> int:
        """
        Process a single CSV file and reconstruct all contained parts using
        a lazy streaming approach so that only one part is materialised in
        memory at any given time.
        """
        reconstructed_count = 0
        print(f"\n📦 Processing: {csv_path.name}")

        try:
            # Lazy scan to avoid loading the full file
            scan = pl.scan_csv(csv_path)

            # Collect distinct part indices (small – just one column)
            unique_parts = (
                scan.select(pl.col("part")).unique().collect()["part"].to_list()
            )
            print(f"   Found {len(unique_parts)} parts")

            group_path = self.extract_group_path_from_filename(csv_path.name)

            for part_index in tqdm(unique_parts, desc="Reconstructing parts"):
                try:
                    filename = self.reconstruct_filename_from_index(part_index)
                    output_path = self.output_dir / group_path / filename

                    # Skip if already exists
                    if self.skip_existing and self.is_file_already_reconstructed(
                        output_path
                    ):
                        print(f"  ⏭️  Skipping existing: {filename}")
                        continue

                    # Materialise data only for this part
                    part_df = (
                        scan.filter(pl.col("part") == part_index)
                        .select(["value", "part"])  # keep only needed cols
                        .collect()
                    )

                    data = self.reconstruct_npy_from_part(part_df, part_index)
                    if data is None:
                        continue

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, data, allow_pickle=False)

                    file_hash = self.calculate_file_hash(output_path)
                    rel_path = output_path.relative_to(self.output_dir)
                    self.reconstructed_hashes[rel_path.as_posix()] = file_hash

                    reconstructed_count += 1
                    self.file_count += 1

                except Exception as e:
                    print(f"  ❌ Failed to process part {part_index}: {e}")

        except Exception as e:
            print(f"❌ Failed to process CSV file {csv_path}: {e}")

        return reconstructed_count

    def process_csv_group(self, csv_files: List[Path], group_name: str) -> int:
        """
        Process a group of CSV chunk files lazily, reconstructing one part at a
        time to keep memory usage bounded.
        """
        reconstructed_count = 0
        print(f"\n📦 Processing group: {group_name}")
        print(f"   Files in group: {[f.name for f in csv_files]}")

        try:
            # Concatenate all chunk scans lazily
            scans = [pl.scan_csv(p) for p in csv_files]
            combined_scan = pl.concat(scans)

            # Determine unique parts without loading full data
            unique_parts = (
                combined_scan.select(pl.col("part"))
                .unique()
                .collect()["part"]
                .to_list()
            )
            print(f"   Found {len(unique_parts)} parts")

            group_path = self.extract_group_path_from_filename(csv_files[0].name)

            for part_index in tqdm(unique_parts, desc="Reconstructing parts"):
                try:
                    filename = self.reconstruct_filename_from_index(part_index)
                    output_path = self.output_dir / group_path / filename

                    if self.skip_existing and self.is_file_already_reconstructed(
                        output_path
                    ):
                        print(f"  ⏭️  Skipping existing: {filename}")
                        continue

                    # Materialise only rows for this part across all chunks
                    part_df = (
                        combined_scan.filter(pl.col("part") == part_index)
                        .select(["value", "part"])
                        .collect()
                    )

                    data = self.reconstruct_npy_from_part(part_df, part_index)
                    if data is None:
                        continue

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, data, allow_pickle=False)

                    file_hash = self.calculate_file_hash(output_path)
                    rel_path = output_path.relative_to(self.output_dir)
                    self.reconstructed_hashes[rel_path.as_posix()] = file_hash

                    reconstructed_count += 1
                    self.file_count += 1

                except Exception as e:
                    print(f"  ❌ Failed to process part {part_index}: {e}")

        except Exception as e:
            print(f"❌ Failed to process CSV group {group_name}: {e}")

        return reconstructed_count

    def save_hashes(self):
        """Save reconstructed file hashes to file_hashes.json."""
        hashes_file = self.output_dir / "file_hashes.json"

        with open(hashes_file, "w") as f:
            json.dump(self.reconstructed_hashes, f, indent=2, sort_keys=True)

        print(
            f"\n📊 Saved {len(self.reconstructed_hashes)} file hashes to: {hashes_file}"
        )

    def create_summary_report(self) -> dict:
        """Create a summary report of the reconstruction."""
        directory_counts: Dict[str, int] = {}
        total_size = 0

        for file_path in self.reconstructed_hashes.keys():
            directory = str(Path(file_path).parent)
            directory_counts[directory] = directory_counts.get(directory, 0) + 1

            full_path = self.output_dir / file_path
            if full_path.exists():
                total_size += full_path.stat().st_size

        report = {
            "reconstruction_summary": {
                "total_files": len(self.reconstructed_hashes),
                "total_size_bytes": total_size,
                "total_size_gb": round(total_size / (1024**3), 2),
                "directories": len(directory_counts),
            },
            "directory_breakdown": directory_counts,
            "sample_files": list(self.reconstructed_hashes.keys())[:10],
        }

        report_file = self.output_dir / "reconstruction_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📋 Reconstruction report saved: {report_file}")
        return report

    def reconstruct_all(self):
        """Reconstruct all CSV files in the input directory."""
        print("🔄 Starting data reconstruction...")
        print(f"📁 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")

        if self.skip_existing:
            print("⏭️  Skip existing: enabled")

        csv_files = list(self.input_dir.glob("*.csv"))
        if not csv_files:
            print("❌ No CSV files found in input directory")
            return

        # Group files by base name (handle chunks)
        file_groups = self.group_csv_files(csv_files)
        print(f"🗂️  Found {len(csv_files)} CSV files in {len(file_groups)} groups")

        total_reconstructed = 0
        for group_name, group_files in file_groups.items():
            count = self.process_csv_group(group_files, group_name)
            total_reconstructed += count
            print(f"   ✅ Reconstructed: {count} files")

        self.save_hashes()
        report = self.create_summary_report()

        print("\n🎉 Reconstruction complete!")
        print(f"   ✅ Total files reconstructed: {total_reconstructed}")
        print("   📁 Files organized in:", self.output_dir)
        print(
            f"   📊 Total size: {report['reconstruction_summary']['total_size_gb']} GB"
        )
        print(
            f"   🗂️  Directories created: {report['reconstruction_summary']['directories']}"
        )
        print("   🔐 Hashes saved: file_hashes.json")

        # Show directory structure sample
        print("\n📋 Directory structure sample:")
        for directory, count in list(report["directory_breakdown"].items())[:5]:
            print(f"   📁 {directory}/ ({count} files)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reconstruct dataset from processed CSV files"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="processed_data",
        help="Directory containing CSV files (default: processed_data)",
    )
    parser.add_argument(
        "--output-dir",
        default="reconstructed_data",
        help="Output directory for reconstructed dataset (default: reconstructed_data)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing files (reconstruct all)",
    )

    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1

    reconstructor = DataReconstructor(
        args.input_dir, args.output_dir, skip_existing=not args.no_skip
    )
    reconstructor.reconstruct_all()

    return 0


if __name__ == "__main__":
    exit(main())
