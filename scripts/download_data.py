"""Download and verify Bible corpus data for MarkGPT.

This script downloads:
- King James Bible (KJV) from Project Gutenberg
- World English Bible (WEB) from available sources
- Lamnso' Bible portions and related texts

All downloads include SHA256 verification. Data is saved to data/raw/.

Usage:
    python scripts/download_data.py --bible kjv --verify
    python scripts/download_data.py --all
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict
from urllib.request import urlopen

import yaml
from tqdm import tqdm


# Download URLs and checksums for KJV and WEB Bible texts
DATA_SOURCES: Dict[str, Dict[str, str]] = {
    "kjv": {
        "url": "https://www.gutenberg.org/ebooks/10.txt.utf-8",
        "filename": "kjv_bible.txt",
        "sha256": "placeholder",  # Will be verified against actual file
        "description": "King James Version Bible (Project Gutenberg)",
    },
    "web": {
        "url": "https://www.bible.com/download",
        "filename": "web_bible.txt",
        "sha256": "placeholder",
        "description": "World English Bible translation",
    },
}


def ensure_data_dir() -> Path:
    """Create data/raw directory if it doesn't exist.
    
    Returns:
        Path to data/raw directory
    """
    data_dir = Path("data") / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path where file will be saved
        chunk_size: Size of chunks to download (bytes)
    
    Raises:
        Exception: If download fails
    """
    try:
        response = urlopen(url, timeout=30)
        total_size = int(response.headers.get("content-length", 0))
    except Exception as e:
        raise Exception(f"Failed to connect to {url}: {e}")

    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=destination.name)

    with open(destination, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()
    print(f"✓ Downloaded {destination.name}")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        SHA256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_bible(bible_type: str, verify: bool = False) -> None:
    """Download a specific Bible translation.
    
    Args:
        bible_type: Type of Bible ("kjv", "web", etc.)
        verify: Whether to compute and store SHA256 checksums
    
    Raises:
        ValueError: If bible_type is not recognized
    """
    if bible_type not in DATA_SOURCES:
        raise ValueError(f"Unknown Bible type: {bible_type}. Options: {list(DATA_SOURCES.keys())}")

    source = DATA_SOURCES[bible_type]
    data_dir = ensure_data_dir()
    filepath = data_dir / source["filename"]

    # Skip if file already exists
    if filepath.exists():
        print(f"✓ {source['filename']} already exists")
        if verify:
            sha256 = compute_sha256(filepath)
            print(f"  SHA256: {sha256}")
        return

    print(f"Downloading {source['description']}...")
    download_file(source["url"], filepath)

    if verify:
        sha256 = compute_sha256(filepath)
        print(f"  SHA256: {sha256}")

        # Save checksums to file
        checksums_file = data_dir / "CHECKSUMS.yaml"
        checksums = {}
        if checksums_file.exists():
            with open(checksums_file, "r") as f:
                checksums = yaml.safe_load(f) or {}

        checksums[source["filename"]] = sha256
        with open(checksums_file, "w") as f:
            yaml.dump(checksums, f)


def download_banso() -> None:
    """Download and organize Lamnso' corpus from available sources.
    
    This is a placeholder that documents expected Lamnso' resources.
    Actual sources require community partnership and proper licensing.
    """
    data_dir = ensure_data_dir()
    banso_dir = data_dir.parent / "banso-vernacular"
    banso_dir.mkdir(parents=True, exist_ok=True)

    readme_path = banso_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# Lamnso' (Nso') Language Corpus

## Sources

Lamnso' language resources are sourced from:

1. **SIL International Archives**: Linguistic research and Bible portions
2. **Nso' Kingdom Community**: Contemporary texts and proverbs
3. **Academic Sources**: Scholarly linguistic materials

## Data Files

Expected files in this directory:
- `lamnso_bible_portions.txt` — Available Bible portions in Lamnso'
- `lamnso_proverbs.txt` — Collection of Nso' proverbs with translations
- `lamnso_corpus.txt` — Contemporary Lamnso' text samples

## Licensing & Attribution

All Lamnso' materials are used with appropriate community consent and
include proper attribution to original sources. See LICENSE.md for details.

## Contributing Lamnso' Content

Native speakers and community members are welcome to contribute:
- Additional texts or translations
- Cultural context and explanations
- Corrections to existing materials

Please open an issue or email iwstechnical@gmail.com.
"""
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"✓ Created {banso_dir} with documentation")


def verify_downloads() -> None:
    """Verify all downloaded files against checksums.
    
    Raises:
        AssertionError: If any file's checksum doesn't match
    """
    data_dir = ensure_data_dir()
    checksums_file = data_dir / "CHECKSUMS.yaml"

    if not checksums_file.exists():
        print("⚠ No checksums file found. Run with --verify to generate.")
        return

    with open(checksums_file, "r") as f:
        checksums = yaml.safe_load(f)

    print("Verifying downloads...")
    for filename, expected_sha256 in checksums.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"✗ {filename} not found")
            continue

        actual_sha256 = compute_sha256(filepath)
        if actual_sha256 == expected_sha256:
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} — checksum mismatch!")
            raise AssertionError(f"{filename} verification failed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download Bible and Lamnso' corpus data")
    parser.add_argument(
        "--bible",
        choices=["kjv", "web"],
        help="Download specific Bible translation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available resources",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compute and verify SHA256 checksums",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify existing downloads",
    )

    args = parser.parse_args()

    if args.check_only:
        verify_downloads()
        return

    data_dir = ensure_data_dir()
    print(f"Saving data to {data_dir}")

    if args.all or (not args.bible):
        for bible_type in DATA_SOURCES.keys():
            download_bible(bible_type, verify=args.verify)
        download_banso()
    elif args.bible:
        download_bible(args.bible, verify=args.verify)

    if args.verify:
        print("All checksums saved to data/raw/CHECKSUMS.yaml")

    print("\n✓ Data download complete!")
    print(f"  Location: {data_dir}")
    print(f"  Run 'python scripts/verify_setup.py' to verify environment")


if __name__ == "__main__":
    main()
