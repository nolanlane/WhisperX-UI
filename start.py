#!/usr/bin/env python3
"""
Universal Media Studio - Startup Validation Script
Validates all dependencies before starting the application.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path


# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Startup")

# Paths - adjust for local vs Docker
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent
# Storage directory for persistence (prefer /workspace on RunPod)
STORAGE_DIR = Path("/workspace") if Path("/workspace").exists() else BASE_DIR

MODELS_DIR = STORAGE_DIR / "models"
BIN_DIR = STORAGE_DIR / "bin"

REQUIRED_BINARIES = [
    ("ffmpeg", ["ffmpeg", "-version"]),
    ("ffprobe", ["ffprobe", "-version"]),
]

OPTIONAL_BINARIES = [
]

REQUIRED_PATHS = [
]


def _run(cmd, *, cwd: Path | None = None, timeout: int | None = None):
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        timeout=timeout,
    )


def check_binary(name, cmd):
    """Check if a binary is available and working."""
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def _is_dir_nonempty(path: Path) -> bool:
    try:
        return path.exists() and any(path.iterdir())
    except Exception:
        return False


def ensure_models_downloaded():
    """Option 2 (slim image): download models on first run if missing."""
    whisper_dir = MODELS_DIR / "whisper"

    if _is_dir_nonempty(whisper_dir):
        logger.info("  ✓ Models already present in /workspace; skipping download")
        return

    logger.info("  ⏬ Models not found; downloading on first run. This may take 5-10 minutes...")
    logger.info("     Please check the container logs for detailed progress.")

    downloader = BASE_DIR / "download_models.py"
    if not downloader.exists():
        raise RuntimeError(f"download_models.py not found at {downloader}")

    # Run directly to stdout so the user sees progress in the container logs
    proc = subprocess.run(
        [sys.executable, str(downloader)],
        cwd=str(BASE_DIR),
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError("Model download failed! Check the logs above for details.")

    if not _is_dir_nonempty(whisper_dir):
        raise RuntimeError(
            "Model download script completed but Whisper models directory is still empty."
        )

    logger.info("  ✓ Model download complete")


def main():
    """Main validation and startup routine."""
    # Ensure sitecustomize.py is loaded for the downloader and the app
    os.environ["PYTHONPATH"] = f"{BASE_DIR}:{os.environ.get('PYTHONPATH', '')}"
    
    logger.info("=" * 55)
    logger.info("  Universal Media Studio - Startup Validation")
    logger.info("=" * 55)
    
    errors = []
    warnings = []
    
    # Check required binaries
    logger.info("[Binaries]")
    for name, cmd in REQUIRED_BINARIES:
        if check_binary(name, cmd):
            logger.info(f"  ✓ {name}")
        else:
            logger.error(f"  ✗ {name} - REQUIRED")
            errors.append(name)
    
    # Summary
    logger.info("=" * 55)
    
    if errors:
        logger.error(f"FATAL: Missing required dependencies: {errors}")
        logger.error("Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Option 2: download models on first run if missing
    try:
        ensure_models_downloaded()
    except Exception as e:
        logger.error(f"FATAL: {e}")
        sys.exit(1)

    if warnings:
        logger.warning(f"Warnings (non-fatal): {len(warnings)} items")
        logger.warning("Some features may be limited.")
    else:
        logger.info("All checks passed!")
    
    logger.info("=" * 55)
    logger.info("Starting Universal Media Studio...")
    logger.info("Access the UI at: http://localhost:7860")
    
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
