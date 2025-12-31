#!/usr/bin/env python3
"""
Universal Media Studio - Startup Validation Script
Validates all dependencies before starting the application.
"""

import sys
import subprocess
from pathlib import Path

# Paths - adjust for local vs Docker
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent
BIN_DIR = BASE_DIR / "bin"
CODEFORMER_DIR = BASE_DIR / "CodeFormer"

REQUIRED_BINARIES = [
    ("ffmpeg", ["ffmpeg", "-version"]),
    ("ffprobe", ["ffprobe", "-version"]),
]

OPTIONAL_BINARIES = [
    ("deepFilter", ["deepFilter", "--help"]),
]

REQUIRED_PATHS = [
    (BIN_DIR / "realesrgan-ncnn-vulkan", "Real-ESRGAN binary"),
    (CODEFORMER_DIR / "inference_codeformer.py", "CodeFormer"),
]


def check_binary(name, cmd):
    """Check if a binary is available and working."""
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def main():
    """Main validation and startup routine."""
    print("\n" + "=" * 55)
    print("  Universal Media Studio - Startup Validation")
    print("=" * 55)
    
    errors = []
    warnings = []
    
    # Check required binaries
    print("\n[Binaries]")
    for name, cmd in REQUIRED_BINARIES:
        if check_binary(name, cmd):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - REQUIRED")
            errors.append(name)
    
    # Check optional binaries
    for name, cmd in OPTIONAL_BINARIES:
        if check_binary(name, cmd):
            print(f"  ✓ {name}")
        else:
            print(f"  ⚠ {name} - optional, feature disabled")
            warnings.append(name)
    
    # Check required paths
    print("\n[AI Models & Tools]")
    for path, desc in REQUIRED_PATHS:
        if path.exists():
            print(f"  ✓ {desc}")
        else:
            print(f"  ⚠ {desc} not found at {path}")
            warnings.append(desc)
    
    # Check GPU
    print("\n[GPU]")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ CUDA: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            print("  ⚠ CUDA not available - running in CPU mode (slower)")
            warnings.append("CUDA")
    except Exception as e:
        print(f"  ⚠ GPU check failed: {e}")
        warnings.append("GPU")
    
    # Check NVENC
    print("\n[Hardware Encoding]")
    has_nvenc = False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        if "h264_nvenc" in result.stdout:
            print("  ✓ NVENC hardware encoding available")
            has_nvenc = True
        else:
            print("  ⚠ NVENC not available - using software encoding")
    except Exception:
        print("  ⚠ Could not check NVENC availability")
    
    # Summary
    print("\n" + "=" * 55)
    
    if errors:
        print(f"FATAL: Missing required dependencies: {errors}")
        print("Please install missing dependencies and try again.")
        sys.exit(1)
    
    if warnings:
        print(f"Warnings (non-fatal): {len(warnings)} items")
        print("Some features may be limited.")
    else:
        print("All checks passed!")
    
    print("=" * 55)
    print("\nStarting Universal Media Studio...")
    print("Access the UI at: http://localhost:7860\n")
    
    # Import and run app
    from app import create_ui, APP_TITLE, DEVICE, HAS_NVENC
    
    app = create_ui()
    app.queue(
        max_size=10,
        default_concurrency_limit=1
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )


if __name__ == "__main__":
    main()
