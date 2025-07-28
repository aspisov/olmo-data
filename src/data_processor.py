#!/usr/bin/env python3
"""
Data Processing Pipeline

Downloads .npy files from URLs, converts to CSV with part tracking,
and saves as single CSV files per dataset group.

Usage:
    python src/data_processor.py config.yaml
"""

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
import gc
import numpy as np
import polars as pl
import requests
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm


class DataProcessor:
    """Handles downloading, conversion, and packaging of dataset files."""

    # Maximum CSV rows to prevent transfer issues (safe limit under 2^31-1)
    MAX_CSV_ROWS = 2_000_000_000
    MAX_CSV_BYTES = 40 * 1024**3

    def __init__(
        self,
        config_path: str,
        output_dir: str = "processed_data",
        upload_to_hf: bool = False,
        hf_repo: str = "aspisov/dataset",
        skip_existing: bool = True,
        delete_after_upload: bool = False,
    ):
        """
        Initialize the data processor.

        Args:
            config_path (str): Path to YAML config file.
            output_dir (str): Directory for output files.
            upload_to_hf (bool): Whether to upload to HuggingFace.
            hf_repo (str): HuggingFace repository ID.
            skip_existing (bool): Whether to skip already processed groups.
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("temp_downloads")
        self.hashes_file = self.output_dir / "file_hashes.json"
        self.upload_to_hf = upload_to_hf
        self.hf_repo = hf_repo
        self.skip_existing = skip_existing
        self.delete_after_upload = delete_after_upload

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Hash storage
        self.file_hashes: Dict[str, str] = {}

        # Progress tracking
        self.progress_file = self.output_dir / "processing_progress.json"
        self.progress_data: Dict[str, Dict] = {}

        # Load existing progress and hashes
        self._load_existing_progress()
        self._load_existing_hashes()

        # Initialize HuggingFace API if needed
        if self.upload_to_hf:
            self._setup_huggingface()

    def _setup_huggingface(self):
        """Setup HuggingFace API connection."""
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("❌ HF_TOKEN not found in .env file for HuggingFace upload")
            self.upload_to_hf = False
        else:
            self.hf_api = HfApi(token=hf_token)
            print(f"🔑 HuggingFace upload enabled to: {self.hf_repo}")

    def _get_csv_path(self, group_name: str) -> Path:
        """Get the CSV file path for a dataset group."""
        encoded_name = group_name.replace("/", "__").replace(":", "").rstrip("_")
        return self.output_dir / f"{encoded_name}.csv"

    def _load_existing_progress(self):
        """Load existing processing progress from file."""
        if self.progress_file.exists():
            try:
                import json

                with open(self.progress_file, "r") as f:
                    self.progress_data = json.load(f)
                print(f"📋 Loaded progress from: {self.progress_file}")
                print(f"   Found progress for {len(self.progress_data)} groups")
            except Exception as e:
                print(f"⚠️  Failed to load progress file: {e}")
                self.progress_data = {}
        else:
            print(f"📋 No existing progress file found, starting fresh")
            self.progress_data = {}

    def _save_progress(self):
        """Save current processing progress to file."""
        try:
            import json

            with open(self.progress_file, "w") as f:
                json.dump(self.progress_data, f, indent=2)
            print(f"💾 Saved progress to: {self.progress_file}")
        except Exception as e:
            print(f"❌ Failed to save progress: {e}")

    def _load_existing_hashes(self):
        """Load existing file hashes (append mode)."""
        if self.hashes_file.exists():
            try:
                import json

                with open(self.hashes_file, "r") as f:
                    existing_hashes = json.load(f)
                    self.file_hashes.update(existing_hashes)
                print(f"🔐 Loaded {len(existing_hashes)} existing hashes")
            except Exception as e:
                print(f"⚠️  Failed to load existing hashes: {e}")

    def _get_group_progress(self, group_name: str) -> Dict:
        """Get progress data for a specific group."""
        if group_name not in self.progress_data:
            self.progress_data[group_name] = {
                "processed_files": [],
                "current_chunk_index": 0,
                "current_chunk_rows": 0,
                "current_chunk_bytes": 0,
                "total_rows_processed": 0,
                "total_parts_processed": 0,
                "chunks_saved": [],
                "last_saved_file": None,
                "status": "in_progress",
            }
        return self.progress_data[group_name]

    def _update_group_progress(self, group_name: str, **updates):
        """Update progress data for a specific group."""
        progress = self._get_group_progress(group_name)
        progress.update(updates)
        self._save_progress()

    def _mark_file_processed(self, group_name: str, filename: str):
         """Record that *filename* is now processed (but don’t touch last_saved_file)."""
         progress = self._get_group_progress(group_name)
         if filename not in progress["processed_files"]:
             progress["processed_files"].append(filename)
             self._save_progress()

    def _is_file_already_processed(self, group_name: str, filename: str) -> bool:
        """Check if a file has already been processed."""
        progress = self._get_group_progress(group_name)
        return filename in progress["processed_files"]

    def _get_unprocessed_urls(self, group_name: str, urls: List[str]) -> List[str]:
        """Filter URLs to only include unprocessed files."""
        unprocessed = []
        for url in urls:
            filename = Path(urlparse(url).path).name
            if not self._is_file_already_processed(group_name, filename):
                unprocessed.append(url)
        return unprocessed

    def _mark_group_complete(self, group_name: str):
        """Mark a group as completely processed."""
        self._update_group_progress(group_name, status="complete")
        print(f"✅ Marked group as complete: {group_name}")

    def _is_group_processed(self, group_name: str) -> bool:
        """
        Decide whether we can skip this group.

        Strategy:
        1.  If the progress file already says 'complete', trust it.
        2.  Otherwise, compare how many files we've processed with how
            many URLs are in the config.
        3.  Only fall back to the expensive on‑disk chunk scan when we
            are *not* deleting local files.
        """
        progress = self._get_group_progress(group_name)

        # 1️⃣ explicit flag
        if progress.get("status") == "complete":
            return True

        # 2️⃣ file‑count heuristic
        expected = len(self.config["paths"].get(group_name, []))
        if len(progress.get("processed_files", [])) >= expected:
            return True

        # 3️⃣ legacy check (only works when files are still local)
        if not self.delete_after_upload:
            is_complete, *_ = self._check_group_completeness(group_name, expected)
            return is_complete

        return False

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from URL with progress bar.

        Args:
            url (str): URL to download from.
            output_path (Path): Where to save the file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(output_path, "wb") as f,
                tqdm(
                    desc=f"Downloading {output_path.name}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            return True

        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return False

    def load_binary_data(self, file_path: Path) -> np.ndarray:
        """Load binary .npy data, handling different formats."""
        try:
            return np.load(file_path)
        except Exception:
            try:
                return np.load(file_path, allow_pickle=True)
            except Exception:
                # Try as raw binary data
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                return np.frombuffer(raw_data, dtype="uint32")

    def extract_part_index(self, filename: str) -> str:
        """
        Extract part index from filename.

        Args:
            filename (str): Filename like "part-00-00000.npy" or "part-4-00000.npy"

        Returns:
            str: Part index like "00" or "4"
        """
        name = filename.replace(".npy", "")

        if name.startswith("part-") and "-00000" in name:
            start = len("part-")
            end = name.find("-00000")
            if end > start:
                return name[start:end]

        return filename

    def extract_dataset_path(self, url: str) -> str:
        """
        Extract the dataset path between 'preprocessed' and 'part-' from URL.
        
        Args:
            url (str): URL like 'http://olmo-data.org/preprocessed/tulu_flan/.../part-07-00000.npy'
            
        Returns:
            str: Dataset path like 'tulu_flan/v1-FULLDECON-HARD-TRAIN-60M-shots_all-upweight_1-dialog_false-sep_rulebased/allenai/dolma2-tokenizer/'
        """
        try:
            # Find the 'preprocessed' part
            preprocessed_idx = url.find('/preprocessed/')
            if preprocessed_idx == -1:
                # Fallback to filename if pattern not found
                return Path(urlparse(url).path).name.replace('.npy', '')
            
            # Extract everything after '/preprocessed/'
            after_preprocessed = url[preprocessed_idx + len('/preprocessed/'):]
            
            # Find the last occurrence of 'part-' to split there
            part_idx = after_preprocessed.rfind('/part-')
            if part_idx == -1:
                # If no 'part-' found, use the whole path
                return after_preprocessed.rstrip('/')
            
            # Extract path up to 'part-'
            dataset_path = after_preprocessed[:part_idx + 1]  # Include trailing slash
            
            return dataset_path
            
        except Exception as e:
            print(f"⚠️  Failed to extract dataset path from {url}: {e}")
            # Fallback to filename
            return Path(urlparse(url).path).name.replace('.npy', '')

    def convert_npy_to_dataframe(
        self, npy_file: Path, part_name: str, dataset_path: str = ""
    ) -> pl.DataFrame | None:
        """
        Convert .npy file to DataFrame with part index and dataset path columns.

        Args:
            npy_file (Path): Input .npy file.
            part_name (str): Name of the part (original filename).
            dataset_path (str): Dataset path extracted from URL.

        Returns:
            pl.DataFrame | None: DataFrame with part index and dataset path columns, or None if failed.
        """
        try:
            data = self.load_binary_data(npy_file)
            part_index = self.extract_part_index(part_name)

            if data.ndim == 1:
                df = pl.DataFrame({
                    "value": data, 
                    "part": [part_index] * len(data),
                    "dataset_path": [dataset_path] * len(data)
                })
                print(
                    f"  ✅ Converted: {data.shape} → {len(df)} rows (part: {part_index}, path: {dataset_path})"
                )

                del data
                gc.collect()

                return df
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")

        except Exception as e:
            print(f"  ❌ Conversion failed for {part_name}: {e}")
            return None

    def save_chunked_csv(
        self, combined_df: pl.DataFrame, base_csv_path: Path, group_name: str
    ) -> List[Path]:
        """
        Save DataFrame as chunked CSV files if too large.

        Args:
            combined_df (pl.DataFrame): DataFrame to save.
            base_csv_path (Path): Base path for CSV files.
            group_name (str): Name of the dataset group.

        Returns:
            List[Path]: List of saved CSV file paths.
        """
        total_rows = len(combined_df)
        saved_files = []

        if total_rows <= self.MAX_CSV_ROWS:
            # Single file - no chunking needed
            combined_df.write_csv(base_csv_path)
            saved_files.append(base_csv_path)
            print(f"   CSV created: {base_csv_path}")
        else:
            # Split into chunks
            num_chunks = (
                total_rows + self.MAX_CSV_ROWS - 1
            ) // self.MAX_CSV_ROWS  # Ceiling division
            print(
                f"   🔀 Splitting into {num_chunks} chunks (total rows: {total_rows:,})"
            )

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * self.MAX_CSV_ROWS
                end_idx = min((chunk_idx + 1) * self.MAX_CSV_ROWS, total_rows)

                chunk_df = combined_df.slice(start_idx, end_idx - start_idx)

                if chunk_idx == 0:
                    # First chunk keeps original name
                    chunk_path = base_csv_path
                else:
                    # Subsequent chunks get _chunk_N suffix
                    chunk_path = base_csv_path.with_stem(
                        f"{base_csv_path.stem}_chunk_{chunk_idx}"
                    )

                chunk_df.write_csv(chunk_path)
                saved_files.append(chunk_path)
                print(
                    f"   📝 Chunk {chunk_idx + 1}/{num_chunks}: {chunk_path.name} ({len(chunk_df):,} rows)"
                )

        return saved_files

    def _get_existing_chunks(self, group_name: str) -> List[Path]:
        """Get all existing chunk files for a dataset group."""
        base_csv_path = self._get_csv_path(group_name)
        base_pattern = base_csv_path.stem

        existing_chunks = []

        # Check for main file (first chunk)
        if base_csv_path.exists():
            existing_chunks.append(base_csv_path)

        # Check for additional chunks
        chunk_idx = 1
        while True:
            chunk_path = base_csv_path.with_stem(f"{base_pattern}_chunk_{chunk_idx}")
            if chunk_path.exists():
                existing_chunks.append(chunk_path)
                chunk_idx += 1
            else:
                break

        return existing_chunks

    def _check_group_completeness(
        self, group_name: str, expected_source_files: int
    ) -> tuple[bool, List[Path], int]:
        """
        Check if a dataset group processing is complete by examining existing chunks.

        Args:
            group_name (str): Name of the dataset group.
            expected_source_files (int): Number of source .npy files expected.

        Returns:
            tuple: (is_complete, existing_chunk_paths, total_rows_in_chunks)
        """
        existing_chunks = self._get_existing_chunks(group_name)

        if not existing_chunks:
            return False, [], 0

        # Count total rows in existing chunks
        total_rows = 0
        unique_parts = set()

        try:
            print(f"   🔍 Checking existing chunks: {len(existing_chunks)} files")

            for chunk_path in existing_chunks:
                # Count rows efficiently using polars
                chunk_rows = pl.scan_csv(chunk_path).select(pl.len()).collect().item()
                total_rows += chunk_rows

                # Also check unique parts to see processing progress
                parts = (
                    pl.scan_csv(chunk_path)
                    .select("part")
                    .unique()
                    .collect()["part"]
                    .to_list()
                )
                unique_parts.update(parts)

                print(f"     📄 {chunk_path.name}: {chunk_rows:,} rows")

            print(f"   📊 Total: {total_rows:,} rows, {len(unique_parts)} unique parts")
            print(f"   🎯 Expected source files: {expected_source_files}")

            # Heuristic: If we have as many unique parts as expected source files,
            # and we have substantial data, consider it complete
            is_complete = len(unique_parts) >= expected_source_files and total_rows > 0

            if is_complete:
                print(
                    f"   ✅ Group appears complete ({len(unique_parts)}/{expected_source_files} parts processed)"
                )
            else:
                print(
                    f"   ⚠️  Group appears incomplete ({len(unique_parts)}/{expected_source_files} parts processed)"
                )

            return is_complete, existing_chunks, total_rows

        except Exception as e:
            print(f"   ❌ Error checking chunks: {e}")
            # If we can't read the chunks, assume incomplete
            return False, existing_chunks, 0

    def process_dataset_group(self, group_name: str, urls: List[str]) -> bool:
        """
        Process a complete dataset group with progress tracking and resumption.

        Args:
            group_name (str): Name of the dataset group.
            urls (List[str]): List of URLs to process.

        Returns:
            bool: True if successful, False otherwise.
        """
        print(f"\n📦 Processing group: {group_name}")

        # Get current progress
        progress = self._get_group_progress(group_name)

        # Check if already complete
        if progress.get("status") == "complete" and self.skip_existing:
            print(f"⏭️  Skipping already completed group: {group_name}")
            print(f"   Total rows: {progress.get('total_rows_processed', 0):,}")
            print(f"   Chunks: {len(progress.get('chunks_saved', []))}")
            return True

        # Filter to unprocessed files
        unprocessed_urls = self._get_unprocessed_urls(group_name, urls)

        if not unprocessed_urls:
            print(f"✅ All files already processed for group: {group_name}")
            self._mark_group_complete(group_name)
            return True

        print(f"   Total files: {len(urls)}")
        print(f"   Already processed: {len(urls) - len(unprocessed_urls)}")
        print(f"   Remaining: {len(unprocessed_urls)}")

        if len(urls) - len(unprocessed_urls) > 0:
            print(
                f"🔄 Resuming from file {len(urls) - len(unprocessed_urls) + 1}/{len(urls)}"
            )

        csv_path = self._get_csv_path(group_name)

        # Resume chunk state from progress
        current_chunk_data: List[pl.DataFrame] = []
        current_chunk_files: List[str] = []  # Track files in current chunk
        current_chunk_rows = progress.get("current_chunk_rows", 0)
        current_chunk_bytes = progress.get("current_chunk_bytes", 0)
        chunk_index = progress.get("current_chunk_index", 0)
        saved_files: List[Path] = []
        total_rows_processed = progress.get("total_rows_processed", 0)
        total_parts_processed = progress.get("total_parts_processed", 0)

        # Restore saved chunk paths
        for chunk_name in progress.get("chunks_saved", []):
            chunk_path = self.output_dir / chunk_name
            if chunk_path.exists():
                saved_files.append(chunk_path)

        def save_current_chunk():
            """Save the current chunk buffer to disk."""
            nonlocal \
                current_chunk_data, \
                current_chunk_files, \
                current_chunk_rows, \
                chunk_index, \
                saved_files

            if not current_chunk_data:
                return

            # Combine current chunk data
            chunk_df = pl.concat(current_chunk_data, how="vertical")

            # Determine chunk file path
            if chunk_index == 0:
                chunk_path = csv_path
            else:
                chunk_path = csv_path.with_stem(f"{csv_path.stem}_chunk_{chunk_index}")

            # Save chunk
            chunk_df.write_csv(chunk_path)
            saved_files.append(chunk_path)

            print(
                f"   💾 Saved chunk {chunk_index + 1}: {chunk_path.name} ({len(chunk_df):,} rows)"
            )

            # ONLY NOW mark files as processed (after successful save)
            for filename in current_chunk_files:
                self._mark_file_processed(group_name, filename)

            if current_chunk_files:
                self._update_group_progress(
                    group_name, last_saved_file=current_chunk_files[-1]
                )
                print(f"     ✅ Marked {len(current_chunk_files)} files as processed")

            # Update progress
            chunks_saved = [f.name for f in saved_files]
            self._update_group_progress(
                group_name,
                current_chunk_index=chunk_index + 1,
                current_chunk_rows=0,
                current_chunk_bytes=0,
                chunks_saved=chunks_saved,
            )

            # Upload to HuggingFace immediately if enabled
            if self.upload_to_hf:
                self.upload_to_huggingface(
                    chunk_path, f"Add processed dataset group: {group_name}"
                )

            # Reset chunk buffer
            current_chunk_data = []
            current_chunk_files = []  # Reset files list too
            current_chunk_rows = 0
            current_chunk_bytes = 0
            chunk_index += 1

        try:
            for url in tqdm(unprocessed_urls, desc="Processing files"):
                url_path = Path(urlparse(url).path)
                filename = url_path.name
                temp_npy = self.temp_dir / filename

                try:
                    # Download with retry
                    while not self.download_file(url, temp_npy):
                        print(f"  ❌ Failed to download {url}, retrying...")
                        time.sleep(1)

                    # Calculate hash and store with extracted dataset path
                    file_hash = self.calculate_file_hash(temp_npy)
                    dataset_path = self.extract_dataset_path(url)
                    self.file_hashes[f"{dataset_path}{filename}"] = file_hash

                    # Convert to DataFrame with dataset path
                    df = self.convert_npy_to_dataframe(temp_npy, filename, dataset_path)
                    if df is not None:
                        df_rows = len(df)
                        df_bytes = df.estimated_size()

                        # Check if adding this DataFrame would exceed chunk limit
                        if ((
                            current_chunk_rows + df_rows > self.MAX_CSV_ROWS or current_chunk_bytes + df_bytes > self.MAX_CSV_BYTES
                        ) and current_chunk_data):
                            # Save current chunk before adding new data
                            save_current_chunk()

                        # Add to current chunk
                        current_chunk_data.append(df)
                        current_chunk_files.append(filename)  # Track file in chunk
                        current_chunk_rows += df_rows
                        current_chunk_bytes += df_bytes
                        total_rows_processed += df_rows
                        total_parts_processed += 1

                        print(f"  ✅ Processed: {filename} ({df_rows:,} rows)")
                        print(
                            f"     Current chunk: {current_chunk_rows:,}/{self.MAX_CSV_ROWS:,} rows"
                            f"({current_chunk_bytes / 1024**3:.1f}/{self.MAX_CSV_BYTES / 1024**3:.0f} GiB)"
                        )
                        print(f"     Files in chunk: {len(current_chunk_files)}")

                        # Update progress immediately
                        self._update_group_progress(
                            group_name,
                            current_chunk_rows=current_chunk_rows,
                            current_chunk_bytes=current_chunk_bytes,
                            total_rows_processed=total_rows_processed,
                            total_parts_processed=total_parts_processed,
                        )

                    # NOTE: File marked as processed only when chunk is saved!

                    # Cleanup
                    temp_npy.unlink()

                except Exception as e:
                    print(f"  ❌ Failed to process {filename}: {e}")
                    if temp_npy.exists():
                        temp_npy.unlink()
                    continue

            # Save final chunk if there's remaining data
            if current_chunk_data:
                save_current_chunk()

            if saved_files:
                print(f"✅ Completed group: {group_name}")
                print(f"   Total rows: {total_rows_processed:,}")
                print(f"   Parts included: {total_parts_processed}")
                print(f"   Files created: {len(saved_files)}")

                # Mark as complete
                self._mark_group_complete(group_name)

                # Save hashes after each group (append mode)
                self.save_hashes()

                return True
            else:
                print(f"❌ No data frames to save for group: {group_name}")
                return False

        except Exception as e:
            print(f"❌ Failed to process group {group_name}: {e}")
            return False

    def upload_to_huggingface(self, file_path: Path, commit_message: str | None = None):
        """Upload a file to HuggingFace dataset repository in processed_data folder."""
        if not self.upload_to_hf:
            return

        try:
            if not file_path.exists():
                print(f"❌ File not found for upload: {file_path}")
                return

            file_size = file_path.stat().st_size
            if file_size == 0:
                print(f"❌ File is empty, skipping upload: {file_path}")
                return

            # Upload to processed_data folder in the repository
            repo_path = f"general/{file_path.name}"
            
            print(
                f"📤 Uploading {file_path.name} ({file_size / (1024 * 1024):.1f} MB) to data/ in HuggingFace..."
            )

            self.hf_api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_path,
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=commit_message
                or f"Add processed dataset: {file_path.name}",
            )

            print(f"✅ Uploaded: {file_path.name} → {repo_path}")

            # Delete file after successful upload if enabled
            if self.delete_after_upload:
                file_path.unlink()
                print(f"🗑️  Deleted local file: {file_path.name}")

        except Exception as e:
            print(f"❌ Failed to upload {file_path.name}: {e}")

    def save_hashes(self):
        """Save all file hashes to JSON file (append mode)."""
        import json

        with open(self.hashes_file, "w") as f:
            json.dump(self.file_hashes, f, indent=2)
        print(f"💾 Saved {len(self.file_hashes)} hashes to: {self.hashes_file}")

        if self.upload_to_hf:
            self.upload_to_huggingface(self.hashes_file, "Update file hashes")

    def process_all(self):
        """Process all dataset groups from config."""
        print("🚀 Starting data processing pipeline...")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🗂️  Found {len(self.config['paths'])} dataset groups")

        if self.skip_existing:
            print("⏭️  Skip existing: enabled")

        successful_groups = 0
        skipped_groups = 0
        total_groups = len(self.config["paths"])

        for group_name, urls in self.config["paths"].items():
            if self.skip_existing and self._is_group_processed(group_name):
                skipped_groups += 1
                print(f"\n⏭️  Skipping: {group_name} (already exists)")
                continue

            if self.process_dataset_group(group_name, urls):
                successful_groups += 1

            # Save hashes after each group
            self.save_hashes()

        # Cleanup
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        print("\n🎉 Processing complete!")
        print(f"   ✅ Successful: {successful_groups} groups")
        print(f"   ⏭️  Skipped: {skipped_groups} groups")
        print(
            f"   📊 Total: {successful_groups + skipped_groups}/{total_groups} groups"
        )
        print(f"   📁 Output location: {self.output_dir}")
        print(f"   🔐 Hashes saved: {self.hashes_file}")


def main():
    """Main entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process dataset files from URLs")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--upload-hf", action="store_true", help="Upload to HuggingFace"
    )
    parser.add_argument(
        "--hf-repo", default="aspisov/dataset", help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--output-dir", default="processed_data", help="Output directory"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing files (reprocess all)",
    )
    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="Delete local files after successful upload to HuggingFace",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    processor = DataProcessor(
        config_path=args.config,
        output_dir=args.output_dir,
        upload_to_hf=args.upload_hf,
        hf_repo=args.hf_repo,
        skip_existing=not args.no_skip,
    )
    processor.process_all()


if __name__ == "__main__":
    main()
