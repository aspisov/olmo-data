#!/usr/bin/env python3
"""
Upload files to HuggingFace that aren't already there.

This script scans a local folder and uploads files to a HuggingFace dataset repository,
skipping files that already exist in the repository.

Usage:
    python scripts/upload_to_hf.py /path/to/local/folder --repo-id username/repo-name
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm


class HFUploader:
    """Upload files to HuggingFace dataset repository, skipping existing files."""

    def __init__(self, repo_id: str, repo_type: str = "dataset"):
        """
        Initialize the HuggingFace uploader.

        Args:
            repo_id (str): HuggingFace repository ID (e.g., "username/repo-name").
            repo_type (str): Repository type ("dataset" or "model").
        """
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._setup_huggingface()

    def _setup_huggingface(self):
        """Setup HuggingFace API connection."""
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("❌ HF_TOKEN not found in .env file")
            print("   Please set HF_TOKEN in your .env file or environment")
            sys.exit(1)
        
        self.hf_api = HfApi(token=hf_token)
        print(f"🔑 Connected to HuggingFace: {self.repo_id}")

    def get_existing_files(self) -> Set[str]:
        """
        Get list of files already in the HuggingFace repository.

        Returns:
            Set[str]: Set of filenames already in the repository.
        """
        try:
            files_info = self.hf_api.list_repo_files(
                repo_id=self.repo_id, 
                repo_type=self.repo_type
            )
            return set(files_info)
        except Exception as e:
            print(f"❌ Failed to get existing files: {e}")
            return set()

    def upload_file(self, file_path: Path, repo_path: str | None = None) -> bool:
        """
        Upload a single file to HuggingFace.

        Args:
            file_path (Path): Local file path to upload.
            repo_path (str | None): Path in repository (defaults to filename).

        Returns:
            bool: True if upload successful, False otherwise.
        """
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False

        if file_path.stat().st_size == 0:
            print(f"⚠️  Skipping empty file: {file_path}")
            return False

        # Use filename as repo path if not specified
        if repo_path is None:
            repo_path = file_path.name

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        try:
            print(f"📤 Uploading {file_path.name} ({file_size_mb:.1f} MB) to {repo_path}...")
            
            self.hf_api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=f"Add file: {file_path.name}"
            )
            
            print(f"✅ Uploaded: {file_path.name} → {repo_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload {file_path.name}: {e}")
            return False

    def upload_folder(
        self, 
        folder_path: Path, 
        repo_folder: str = "", 
        file_pattern: str = "*",
        recursive: bool = False
    ) -> tuple[int, int]:
        """
        Upload all files from a folder to HuggingFace.

        Args:
            folder_path (Path): Local folder path to scan.
            repo_folder (str): Folder path in repository (e.g., "data/").
            file_pattern (str): Glob pattern for files to upload.
            recursive (bool): Whether to scan subdirectories recursively.

        Returns:
            tuple[int, int]: (uploaded_count, skipped_count)
        """
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return 0, 0

        if not folder_path.is_dir():
            print(f"❌ Path is not a directory: {folder_path}")
            return 0, 0

        print(f"�� Scanning folder: {folder_path}")
        print(f"�� Repository folder: {repo_folder or 'root'}")
        
        # Get existing files from HF
        existing_files = self.get_existing_files()
        print(f"📋 Found {len(existing_files)} existing files in repository")

        # Find local files
        if recursive:
            local_files = list(folder_path.rglob(file_pattern))
        else:
            local_files = list(folder_path.glob(file_pattern))

        # Filter out directories
        local_files = [f for f in local_files if f.is_file()]
        
        print(f"📁 Found {len(local_files)} local files")

        uploaded_count = 0
        skipped_count = 0

        for file_path in tqdm(local_files, desc="Uploading files"):
            # Determine repository path
            if repo_folder:
                repo_path = f"{repo_folder.rstrip('/')}/{file_path.name}"
            else:
                repo_path = file_path.name

            # Check if file already exists
            if repo_path in existing_files:
                print(f"⏭️  Skipping {file_path.name} (already exists)")
                skipped_count += 1
                continue

            # Upload file
            if self.upload_file(file_path, repo_path):
                uploaded_count += 1
            else:
                skipped_count += 1

        return uploaded_count, skipped_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload files to HuggingFace that aren't already there"
    )
    parser.add_argument(
        "folder", 
        type=Path, 
        help="Local folder path to scan for files"
    )
    parser.add_argument(
        "--repo-id", 
        required=True,
        help="HuggingFace repository ID (e.g., 'username/repo-name')"
    )
    parser.add_argument(
        "--repo-type", 
        default="dataset",
        choices=["dataset", "model"],
        help="Repository type (default: dataset)"
    )
    parser.add_argument(
        "--repo-folder", 
        default="",
        help="Folder path in repository (e.g., 'data/')"
    )
    parser.add_argument(
        "--pattern", 
        default="*",
        help="File pattern to match (default: all files)"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Scan subdirectories recursively"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.folder.exists():
        print(f"❌ Folder not found: {args.folder}")
        sys.exit(1)

    if not args.folder.is_dir():
        print(f"❌ Path is not a directory: {args.folder}")
        sys.exit(1)

    # Initialize uploader
    try:
        uploader = HFUploader(args.repo_id, args.repo_type)
    except Exception as e:
        print(f"❌ Failed to initialize HuggingFace connection: {e}")
        sys.exit(1)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be uploaded")
        existing_files = uploader.get_existing_files()
        
        if args.recursive:
            local_files = list(args.folder.rglob(args.pattern))
        else:
            local_files = list(args.folder.glob(args.pattern))
        
        local_files = [f for f in local_files if f.is_file()]
        
        print(f"\n📁 Found {len(local_files)} local files:")
        for file_path in local_files:
            if args.repo_folder:
                repo_path = f"{args.repo_folder.rstrip('/')}/{file_path.name}"
            else:
                repo_path = file_path.name
            
            status = "⏭️  SKIP" if repo_path in existing_files else "📤 UPLOAD"
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {status} {file_path.name} ({size_mb:.1f} MB)")
        
        return

    # Upload files
    uploaded, skipped = uploader.upload_folder(
        folder_path=args.folder,
        repo_folder=args.repo_folder,
        file_pattern=args.pattern,
        recursive=args.recursive
    )

    # Summary
    print(f"\n🎉 Upload complete!")
    print(f"   ✅ Uploaded: {uploaded} files")
    print(f"   ⏭️  Skipped: {skipped} files")
    print(f"   �� Total: {uploaded + skipped} files processed")


if __name__ == "__main__":
    main()