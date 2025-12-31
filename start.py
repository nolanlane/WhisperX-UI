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
CODEFORMER_DIR = MODELS_DIR / "CodeFormer"

REQUIRED_BINARIES = [
    ("ffmpeg", ["ffmpeg", "-version"]),
    ("ffprobe", ["ffprobe", "-version"]),
]

OPTIONAL_BINARIES = [
    ("df-enhance", ["df-enhance", "--help"]),
]

REQUIRED_PATHS = [
    (BIN_DIR / "realesrgan-ncnn-vulkan", "Real-ESRGAN binary"),
    (BIN_DIR / "models", "Real-ESRGAN models"),
    (CODEFORMER_DIR / "inference_codeformer.py", "CodeFormer"),
]


def _run(cmd, *, cwd: Path | None = None, timeout: int | None = None):
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        timeout=timeout,
    )


def ensure_realesrgan_binary(base_dir: Path) -> None:
    target = base_dir / "bin" / "realesrgan-ncnn-vulkan"
    models_dir = base_dir / "bin" / "models"
    if target.exists() and models_dir.exists():
        return

    bin_dir = target.parent
    bin_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "realesrgan-ncnn-vulkan-v0.2.0-ubuntu.zip"
    url = (
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/"
        + zip_name
    )
    zip_path = bin_dir / zip_name

    logger.info("  ⏬ Downloading Real-ESRGAN binary...")
    try:
        try:
            _run([
                "curl",
                "-L",
                "-f",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "-o",
                str(zip_path),
                url,
            ], timeout=600)
        except Exception:
            _run(["wget", "-q", "-O", str(zip_path), url], timeout=600)
        _run(["unzip", "-o", str(zip_path)], cwd=bin_dir, timeout=600)

        # The zip extracts into a subfolder (e.g. realesrgan-ncnn-vulkan-v0.2.0-ubuntu/)
        # so we need to locate the contents and move them into the bin directory.
        extracted_dirs = [p for p in bin_dir.glob("realesrgan-ncnn-vulkan-*") if p.is_dir()]
        if not extracted_dirs:
            # Fallback: search for the binary specifically
            binaries = list(bin_dir.glob("**/realesrgan-ncnn-vulkan"))
            binaries = [p for p in binaries if p.is_file() and p.name == "realesrgan-ncnn-vulkan"]
            if binaries:
                src_parent = binaries[0].parent
            else:
                raise RuntimeError(f"Real-ESRGAN unzip completed but contents not found in {bin_dir}")
        else:
            src_parent = extracted_dirs[0]

        import shutil
        # Move all contents (binary, models folder, etc.) to bin_dir
        for item in src_parent.iterdir():
            dest = bin_dir / item.name
            if dest.resolve() == item.resolve():
                continue
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink(missing_ok=True)
            shutil.move(str(item), str(dest))

        # Cleanup the now-empty extracted folder
        if src_parent.exists() and src_parent != bin_dir:
            shutil.rmtree(src_parent, ignore_errors=True)

        _run(["chmod", "+x", str(target)], timeout=30)
    finally:
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not target.exists():
        raise RuntimeError(f"Real-ESRGAN binary install failed; expected at {target}")
    logger.info("  ✓ Real-ESRGAN ready")


def ensure_codeformer_repo(base_dir: Path) -> None:
    repo_dir = base_dir / "CodeFormer"
    marker = repo_dir / "inference_codeformer.py"
    if marker.exists():
        return

    logger.info("  ⏬ Cloning CodeFormer repository...")
    try:
        if repo_dir.exists():
            # If the directory exists but is incomplete/corrupt, remove it.
            # This keeps startup idempotent and avoids half-installed repos.
            import shutil

            shutil.rmtree(repo_dir, ignore_errors=True)

        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/sczhou/CodeFormer.git",
                str(repo_dir),
            ],
            timeout=600,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to clone CodeFormer: {e}")

    if not marker.exists():
        raise RuntimeError(
            "CodeFormer clone completed but expected entrypoint is missing: "
            f"{marker}"
        )
    logger.info("  ✓ CodeFormer repo ready")


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
        logger.info("  ✓ Models already present; skipping download")
        return

    logger.info("  ⏬ Models not found; downloading on first run (this may take a while)...")

    downloader = BASE_DIR / "download_models.py"
    if not downloader.exists():
        raise RuntimeError(f"download_models.py not found at {downloader}")

    log_path = STORAGE_DIR / "temp" / "model_download.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            [sys.executable, str(downloader)],
            cwd=str(BASE_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        try:
            tail = ""
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                tail = "".join(lines[-80:])
        except Exception:
            tail = "(failed to read model download log)"
        raise RuntimeError(
            "Model download failed on first run. "
            f"See log at {log_path}.\n--- log tail ---\n{tail}"
        )

    if not _is_dir_nonempty(whisper_dir):
        raise RuntimeError(
            "Model download completed but Whisper models directory is still empty. "
            f"Check {log_path}."
        )

    try:
        log_path.unlink(missing_ok=True)
    except Exception:
        pass
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
    
    # Check optional binaries
    for name, cmd in OPTIONAL_BINARIES:
        if check_binary(name, cmd):
            logger.info(f"  ✓ {name}")
        else:
            logger.warning(f"  ⚠ {name} - optional, feature disabled")
            warnings.append(name)
    
    # Summary
    logger.info("=" * 55)
    
    if errors:
        logger.error(f"FATAL: Missing required dependencies: {errors}")
        logger.error("Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Slim image strategy: acquire heavyweight runtime assets at pod startup.
    try:
        ensure_realesrgan_binary(STORAGE_DIR)
        ensure_codeformer_repo(MODELS_DIR)
    except Exception as e:
        logger.error(f"FATAL: {e}")
        sys.exit(1)

    # Now check required paths after asset acquisition
    logger.info("[AI Models & Tools]")
    for path, desc in REQUIRED_PATHS:
        if path.exists():
            logger.info(f"  ✓ {desc}")
        else:
            logger.warning(f"  ⚠ {desc} not found at {path}")
            warnings.append(desc)

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
