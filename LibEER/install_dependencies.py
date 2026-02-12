"""
Installation Script for LibEER - Emognition Dataset Support
============================================================
This script installs all required dependencies for using the Emognition
dataset loader with the LibEER framework.

Run this script ONCE before using the Emognition dataset loader:
    python install_dependencies.py

Author: Final Year Project Team
Date: February 2026
"""

import subprocess
import sys
import os


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def install_package(package_name, pip_name=None):
    """
    Install a Python package using pip.
    
    Args:
        package_name: Name to display
        pip_name: Name to use with pip (if different from display name)
    """
    if pip_name is None:
        pip_name = package_name
    
    print(f"\n📦 Installing {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"   ✅ {package_name} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Warning: Could not install {package_name}")
        print(f"   Error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return False


def check_package(package_name, import_name=None):
    """
    Check if a package is already installed.
    
    Args:
        package_name: Name to display
        import_name: Name to use for import (if different from package name)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def main():
    """Main installation process."""
    print_header("LibEER DEPENDENCY INSTALLER - EMOGNITION SUPPORT")
    
    print("\n🎯 This script will install all dependencies needed for:")
    print("   - LibEER framework")
    print("   - Emognition dataset loader")
    print("   - EEG signal processing")
    print("   - Deep learning models")
    
    # Define all required packages
    # Format: (display_name, pip_name, import_name)
    packages = [
        # Core scientific computing
        ("NumPy", "numpy", "numpy"),
        ("Pandas", "pandas", "pandas"),
        ("SciPy", "scipy", "scipy"),
        
        # EEG processing
        ("MNE (EEG/MEG analysis)", "mne", "mne"),
        
        # Deep learning
        ("PyTorch", "torch", "torch"),
        ("torchvision", "torchvision", "torchvision"),
        
        # Machine learning
        ("scikit-learn", "scikit-learn", "sklearn"),
        
        # Data format support
        ("mat73 (MATLAB v7.3 support)", "mat73", "mat73"),
        ("xmltodict (XML parsing)", "xmltodict", "xmltodict"),
        
        # Utilities
        ("tqdm (Progress bars)", "tqdm", "tqdm"),
        ("PyYAML", "pyyaml", "yaml"),
        
        # Graph processing (for GNN models)
        ("torch_geometric", "torch-geometric", "torch_geometric"),
    ]
    
    print_header("CHECKING EXISTING PACKAGES")
    
    already_installed = []
    to_install = []
    
    for display_name, pip_name, import_name in packages:
        if check_package(display_name, import_name):
            print(f"✅ {display_name:<40} Already installed")
            already_installed.append(display_name)
        else:
            print(f"❌ {display_name:<40} Not found")
            to_install.append((display_name, pip_name, import_name))
    
    if not to_install:
        print_header("ALL DEPENDENCIES ALREADY INSTALLED")
        print("\n🎉 All required packages are already installed!")
        print("   You're ready to use the Emognition dataset loader!")
        return
    
    print_header("INSTALLING MISSING PACKAGES")
    
    print(f"\n📋 Found {len(to_install)} package(s) to install:")
    for display_name, _, _ in to_install:
        print(f"   - {display_name}")
    
    print("\n⏳ Installation will begin in 3 seconds...")
    print("   Press Ctrl+C to cancel")
    
    try:
        import time
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        return
    
    # Upgrade pip first
    print("\n🔧 Upgrading pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
            stdout=subprocess.DEVNULL
        )
        print("   ✅ pip upgraded successfully!")
    except:
        print("   ⚠️  Could not upgrade pip (continuing anyway)")
    
    # Install packages
    success_count = 0
    failed_packages = []
    
    for display_name, pip_name, import_name in to_install:
        if install_package(display_name, pip_name):
            success_count += 1
        else:
            failed_packages.append(display_name)
    
    # Special handling for torch-geometric (requires PyTorch first)
    if any("torch_geometric" in str(pkg) for pkg in to_install):
        print("\n🔧 Installing torch-geometric with dependencies...")
        try:
            # Install torch-scatter, torch-sparse, torch-cluster
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", 
                 "torch-scatter", "torch-sparse", "torch-cluster", 
                 "-f", "https://data.pyg.org/whl/torch-2.0.0+cpu.html",
                 "--quiet"],
                stdout=subprocess.DEVNULL
            )
            print("   ✅ torch-geometric dependencies installed!")
        except:
            print("   ⚠️  Warning: torch-geometric dependencies may need manual installation")
    
    # Final summary
    print_header("INSTALLATION SUMMARY")
    
    print(f"\n📊 Results:")
    print(f"   Already installed: {len(already_installed)}")
    print(f"   Successfully installed: {success_count}")
    print(f"   Failed: {len(failed_packages)}")
    
    if failed_packages:
        print(f"\n⚠️  Failed packages:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print(f"\n💡 Try installing these manually with:")
        print(f"   pip install {' '.join(failed_packages)}")
    
    if success_count > 0 or len(already_installed) == len(packages):
        print("\n" + "="*80)
        print("  ✅ INSTALLATION COMPLETE!")
        print("="*80)
        print("\n🚀 You can now:")
        print("   1. Test the Emognition loader:")
        print("      python test_emognition_loader.py --dataset_path /path/to/data")
        print("\n   2. Use it in your scripts:")
        print("      from config.setting import Setting")
        print("      from data_utils.load_data import get_data")
        print("\n" + "="*80)
    else:
        print("\n" + "="*80)
        print("  ⚠️  INSTALLATION INCOMPLETE")
        print("="*80)
        print("\n   Please check the errors above and install missing packages manually.")
        print("="*80)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print(f"\n🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3:
        print("   ❌ ERROR: Python 3.x is required!")
        print("   Please upgrade to Python 3.7 or higher")
        return False
    
    if version.minor < 7:
        print("   ⚠️  WARNING: Python 3.7+ is recommended")
        print("   You have Python 3.{version.minor}, some features may not work")
    else:
        print("   ✅ Python version is compatible!")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  LibEER - EMOGNITION DATASET DEPENDENCY INSTALLER")
    print("  " + "="*78)
    print(f"  Date: February 12, 2026")
    print(f"  Platform: {sys.platform}")
    print(f"  Python: {sys.executable}")
    print("="*80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Run installation
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
