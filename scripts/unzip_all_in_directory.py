#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path


def unzip_all_archives(directory_path: Path):
    target_dir = directory_path.resolve()

    if not target_dir.is_dir():
        print(f"Error: Directory not found at {target_dir}")
        return

    zip_files = list(target_dir.rglob("*.zip"))

    if not zip_files:
        print("No zip files found.")
        return

    for zip_filepath in zip_files:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(zip_filepath.parent)
        print(f"Extracted {zip_filepath.name}")

    print(f"Processed {len(zip_files)} archives.")


def main():
    parser = argparse.ArgumentParser(description="Unzip all archives in directory")
    parser.add_argument("directory", type=Path, help="Absolute or relative path to target directory")
    
    args = parser.parse_args()
    unzip_all_archives(args.directory)


if __name__ == "__main__":
    main()
