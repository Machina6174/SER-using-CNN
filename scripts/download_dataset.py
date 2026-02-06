#!/usr/bin/env python3
"""
Download RAVDESS Dataset for Speech Emotion Recognition.

This script downloads the RAVDESS dataset from Zenodo and extracts it
to the data/raw/ directory.

Usage:
    python scripts/download_dataset.py

The RAVDESS dataset contains 2880 audio files from 24 actors (12 male, 12 female),
each performing 8 emotions.

Dataset source: https://zenodo.org/record/1188976
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# RAVDESS Audio-Only Speech files (not song)
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
EXPECTED_SIZE = 215_000_000  # Approximately 215 MB


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file with progress bar."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)


def main():
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Setup paths
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    zip_path = data_dir / "ravdess.zip"
    
    # Check if dataset already exists
    if raw_dir.exists() and any(raw_dir.iterdir()):
        actor_dirs = list(raw_dir.glob("Actor_*"))
        if len(actor_dirs) >= 24:
            print("✅ Dataset already downloaded!")
            print(f"   Found {len(actor_dirs)} actor directories in {raw_dir}")
            return
    
    # Create directories
    data_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("RAVDESS Dataset Downloader")
    print("=" * 60)
    print(f"\nSource: {RAVDESS_URL}")
    print(f"Destination: {raw_dir}")
    print(f"Expected size: ~215 MB\n")
    
    # Download
    print("📥 Downloading RAVDESS dataset...")
    try:
        download_file(RAVDESS_URL, zip_path)
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("   https://zenodo.org/record/1188976")
        print(f"   Extract to: {raw_dir}")
        sys.exit(1)
    
    # Extract
    print("\n📦 Extracting files...")
    try:
        extract_zip(zip_path, raw_dir)
    except zipfile.BadZipFile as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    # Move files from nested directory if needed
    nested_dir = raw_dir / "Audio_Speech_Actors_01-24"
    if nested_dir.exists():
        print("\n🔧 Reorganizing directory structure...")
        for item in nested_dir.iterdir():
            dest = raw_dir / item.name
            if not dest.exists():
                item.rename(dest)
        nested_dir.rmdir()
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    if zip_path.exists():
        zip_path.unlink()
    
    # Verify
    actor_dirs = list(raw_dir.glob("Actor_*"))
    wav_files = list(raw_dir.glob("**/*.wav"))
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)
    print(f"   Actors: {len(actor_dirs)}")
    print(f"   Audio files: {len(wav_files)}")
    print(f"   Location: {raw_dir}")
    
    if len(wav_files) != 1440:
        print(f"\n⚠️  Warning: Expected 1440 files, found {len(wav_files)}")
        print("   (This is normal if you already had some files)")


if __name__ == "__main__":
    main()
