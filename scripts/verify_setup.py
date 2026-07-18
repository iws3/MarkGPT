"""Verify MarkGPT environment setup and dependencies.

This script checks:
- Python version (3.10+)
- PyTorch installation and CUDA/MPS availability
- All required packages installed
- Data directory structure
- Tokenizer importable

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

import yaml


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_check(name: str, status: bool, message: str = "") -> None:
    """Print a check result with color-coding."""
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {name:<40} {message}")


def check_python_version() -> bool:
    """Check Python version is 3.10 or higher.
    
    Returns:
        True if Python version is adequate
    """
    version = sys.version_info
    required = (3, 10)
    is_valid = version >= required
    msg = f"Python {version.major}.{version.minor}.{version.micro}"
    print_check("Python version", is_valid, msg)
    return is_valid


def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch installation and device availability.
    
    Returns:
        Tuple of (is_valid, device_info)
    """
    try:
        import torch

        version = torch.__version__
        device_info = "CPU"

        if torch.cuda.is_available():
            device_info = f"CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "Apple Metal Performance Shaders (MPS)"

        msg = f"v{version} ({device_info})"
        print_check("PyTorch", True, msg)
        return True, device_info

    except ImportError:
        print_check("PyTorch", False, "Not installed")
        return False, "Not available"


def check_packages() -> bool:
    """Check all required Python packages are installed.
    
    Returns:
        True if all packages are present
    """
    required_packages = {
        "numpy": "NumPy",
        "tqdm": "tqdm",
        "pyyaml": "PyYAML",
        "pytest": "pytest",
    }

    optional_packages = {
        "wandb": "wandb (optional)",
        "transformers": "HuggingFace Transformers (optional)",
        "gradio": "Gradio (optional)",
    }

    all_required = True

    for module, name in required_packages.items():
        try:
            __import__(module)
            print_check(f"  {name}", True)
        except ImportError:
            print_check(f"  {name}", False)
            all_required = False

    for module, name in optional_packages.items():
        try:
            __import__(module)
            print_check(f"  {name}", True)
        except ImportError:
            print_check(f"  {name}", False, "(optional)")

    return all_required


def check_tokenizer() -> bool:
    """Check tokenizer module can be imported.
    
    Returns:
        True if tokenizer imports successfully
    """
    try:
        from src.tokenizer import tokenizer
        from src.tokenizer.tokenizer import Tokenizer

        print_check("Tokenizer import", True)
        return True
    except ImportError as e:
        msg = str(e)
        print_check("Tokenizer import", False, msg)
        return False


def check_data_directory() -> bool:
    """Check data directory structure.
    
    Returns:
        True if data directories exist
    """
    directories = [
        Path("data") / "raw",
        Path("data") / "processed",
    ]

    all_exist = True
    for d in directories:
        exists = d.exists()
        print_check(f"  {d}", exists)
        all_exist = all_exist and exists

    return all_exist


def check_model_config() -> bool:
    """Check model configuration files exist and are valid.
    
    Returns:
        True if configs are valid
    """
    configs = [
        Path("configs") / "markgpt_nano.yaml",
        Path("configs") / "markgpt_small.yaml",
    ]

    all_valid = True
    for config_file in configs:
        if not config_file.exists():
            print_check(f"  {config_file.name}", False, "not found")
            all_valid = False
            continue

        try:
            with open(config_file, "r") as f:
                yaml.safe_load(f)
            print_check(f"  {config_file.name}", True)
        except Exception as e:
            print_check(f"  {config_file.name}", False, str(e))
            all_valid = False

    return all_valid


def main() -> int:
    """Run all verification checks.
    
    Returns:
        0 if all checks pass, 1 otherwise
    """
    print_header("MarkGPT Environment Verification")

    checks = [
        ("Python Version", check_python_version()),
    ]

    pytorch_ok, device_info = check_pytorch()
    checks.append(("PyTorch", pytorch_ok))

    print_header("Required Packages")
    packages_ok = check_packages()
    checks.append(("Packages", packages_ok))

    print_header("Tokenizer")
    tokenizer_ok = check_tokenizer()
    checks.append(("Tokenizer", tokenizer_ok))

    print_header("Data Directory")
    data_ok = check_data_directory()
    checks.append(("Data", data_ok))

    print_header("Model Configuration")
    config_ok = check_model_config()
    checks.append(("Configs", config_ok))

    # Summary
    print_header("Summary")
    all_ok = all(status for _, status in checks)

    for name, status in checks:
        print_check(name, status)

    if all_ok:
        print("\n✓ Environment verified! You're ready to train MarkGPT.")
        print(f"  Compute device: {device_info}")
        print(f"  Next: python scripts/download_data.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please address the issues above.")
        print("  See TROUBLESHOOTING.md for help.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
