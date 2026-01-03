# =============================================================================
# Universal Media Studio - Model Downloader
# Downloads all required models at Docker build time for instant RunPod startup.
# =============================================================================

import os
import sys
import hashlib
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
logger = logging.getLogger("Downloader")

# Paths - robust to local vs Docker
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent
# Storage directory for persistence (prefer /workspace on RunPod)
STORAGE_DIR = Path("/workspace") if Path("/workspace").exists() else BASE_DIR

MODELS_DIR = STORAGE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
HF_HOME = MODELS_DIR / "huggingface"
NLTK_HOME = MODELS_DIR / "nltk"
TORCH_HOME = MODELS_DIR / "torch"

# Set HuggingFace, NLTK, and Torch cache locations
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME)
os.environ["NLTK_DATA"] = str(NLTK_HOME)
os.environ["TORCH_HOME"] = str(TORCH_HOME)


def clear_all_model_caches():
    """Clear all possible model cache directories to resolve checksum issues."""
    cache_dirs = [
        WHISPER_DIR,
        MODELS_DIR / "whisper", 
        MODELS_DIR / "alignment",
        HF_HOME,
        NLTK_HOME,
        TORCH_HOME,
        Path.home() / ".cache" / "whisper",
        Path.home() / ".cache" / "huggingface", 
        Path.home() / ".cache" / "whisperx",
        Path.home() / ".cache" / "torch" / "hub",
        Path("/tmp/whisperx")
    ]
    
    logger.info("🧹 Clearing all model cache directories...")
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            logger.info(f"  Removing: {cache_dir}")
            try:
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir, ignore_errors=True)
                else:
                    cache_dir.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"  Failed to remove {cache_dir}: {e}")
    
    # Recreate essential directories
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "alignment").mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    NLTK_HOME.mkdir(parents=True, exist_ok=True)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Cache clearing completed")


# Ensure directories exist
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Ensure directories exist
WHISPER_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)
NLTK_HOME.mkdir(parents=True, exist_ok=True)
TORCH_HOME.mkdir(parents=True, exist_ok=True)


def download_whisper_model():
    """Download WhisperX large-v3 model with retry logic for checksum errors."""
    logger.info("=" * 60)
    logger.info("📥 Downloading Whisper large-v3 model...")
    logger.info("=" * 60)
    
    def _load():
        import whisperx
        return whisperx.load_model(
            "large-v3",
            device="cpu",
            compute_type="int8",
            download_root=str(WHISPER_DIR)
        )

    try:
        import nltk
        # WhisperX uses nltk for sentence segmentation
        logger.info(f"  Downloading NLTK data to {os.environ['NLTK_DATA']}...")
        nltk.download('punkt', download_dir=os.environ["NLTK_DATA"])
        nltk.download('punkt_tab', download_dir=os.environ["NLTK_DATA"])
        
        try:
            model = _load()
            del model
        except Exception as e:
            err_msg = str(e).lower()
            if "sha256" in err_msg or "checksum" in err_msg:
                logger.warning(f"Checksum mismatch for Whisper model. Clearing all caches and retrying...")
                clear_all_model_caches()
                
                # First retry
                try:
                    model = _load()
                    del model
                    logger.info("✅ Whisper model loaded successfully after cache clear")
                except Exception as retry_e:
                    if "sha256" in str(retry_e).lower() or "checksum" in str(retry_e).lower():
                        # Second retry with pause
                        logger.warning("Second checksum failure, trying with fresh download...")
                        import time
                        time.sleep(3)  # Brief pause
                        model = _load()
                        del model
                    else:
                        raise retry_e
            else:
                raise e
        
        logger.info("✅ Whisper large-v3 downloaded successfully!")
        
    except Exception as e:
        logger.warning(f"⚠️ Whisper download error (may still work at runtime): {e}")


def download_alignment_models():
    """Download WhisperX alignment models for common languages."""
    logger.info("=" * 60)
    logger.info("📥 Downloading alignment models...")
    logger.info("=" * 60)
    
    try:
        import whisperx
        import torch
        
        # Download common alignment models to match UI choices
        languages = ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "vi"]
        for lang in languages:
            logger.info(f"  Downloading alignment model for: {lang}")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=lang,
                    device="cpu",
                    model_dir=str(MODELS_DIR / "alignment")
                )
                del model_a
                logger.info(f"  ✅ {lang} alignment model ready")
            except Exception as e:
                logger.warning(f"  ⚠️ {lang} alignment model: {e}")
        
    except Exception as e:
        logger.error(f"⚠️ Alignment model download error: {e}")


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_vad_model():
    """Download Silero VAD model used by WhisperX with checksum resilience."""
    logger.info("=" * 60)
    logger.info("📥 Downloading VAD model...")
    logger.info("=" * 60)
    
    EXPECTED_VAD_HASH = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"
    
    try:
        import torch
        
        # Silero VAD (standard)
        try:
            torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
        except Exception as e:
            logger.warning(f"Silero VAD Hub load issue: {e}")
            # Non-fatal, whisperx might still work or download it itself
                
        logger.info("✅ Silero VAD model check complete")
        
        # WhisperX specific VAD segmentation model
        logger.info("  Verifying WhisperX VAD segmentation model...")

        # Target location expected by WhisperX
        target_path = TORCH_HOME / "whisperx-vad-segmentation.bin"

        # Verify existing file if any
        if target_path.exists():
            current_hash = calculate_sha256(target_path)
            if current_hash == EXPECTED_VAD_HASH:
                logger.info("  ✓ WhisperX VAD model already exists and is valid.")
                return
            else:
                logger.warning(f"  ⚠️ Hash mismatch for existing VAD model: {current_hash}. Re-downloading...")
                target_path.unlink()

        primary_url = "https://whisperx.s3.us-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"
        fallback_url = "https://github.com/m-bain/whisperX/raw/main/whisperx/assets/pytorch_model.bin"

        urls_to_try = [primary_url, fallback_url]
        downloaded = False
        
        for i, url in enumerate(urls_to_try):
            try:
                logger.info(f"  Trying URL {i+1}/{len(urls_to_try)}: {url}")
                # Use curl -L -f to handle redirects and 404s
                result = subprocess.run([
                    "curl", "-L", "-f",
                    "--user-agent", "Mozilla/5.0 (compatible; WhisperX-UI/1.0)",
                    "-o", str(target_path),
                    url
                ], check=True, timeout=300)
                
                # Verify hash immediately after download
                new_hash = calculate_sha256(target_path)
                if new_hash == EXPECTED_VAD_HASH:
                    downloaded = True
                    logger.info(f"  ✅ Downloaded and verified VAD model from URL {i+1}")
                    break
                else:
                    logger.error(f"  ❌ Hash mismatch for downloaded VAD model from URL {i+1}: {new_hash}")
                    target_path.unlink()
            except Exception as e:
                logger.warning(f"  ⚠️ URL {i+1} failed: {e}")
                if target_path.exists():
                    target_path.unlink()
                continue
        
        if not downloaded:
            logger.error("  ❌ All VAD model download attempts failed or failed verification.")
            # We don't raise here, whisperx might still try its own internal download
            # but we've tried our best.
    except Exception as e:
        logger.error(f"⚠️ VAD model download process error: {e}")


def main():
    """Main download orchestrator."""
    logger.info("=" * 60)
    logger.info("🚀 Universal Media Studio - Model Downloader")
    logger.info("=" * 60)
    logger.info(f"📍 Models directory: {MODELS_DIR}")
    logger.info(f"📍 HuggingFace cache: {HF_HOME}")
    
    # Run all downloads
    download_whisper_model()
    download_alignment_models()
    download_vad_model()
    
    logger.info("=" * 60)
    logger.info("✅ Model download complete!")
    logger.info("=" * 60)
    
    # Print cache sizes
    import shutil
    
    def get_dir_size(path):
        if path.exists():
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total / (1024 ** 3)  # GB
        return 0
    
    logger.info("📊 Cache sizes:")
    logger.info(f"  Whisper models: {get_dir_size(WHISPER_DIR):.2f} GB")
    logger.info(f"  HuggingFace cache: {get_dir_size(HF_HOME):.2f} GB")
    logger.info(f"  Total: {get_dir_size(MODELS_DIR):.2f} GB")


if __name__ == "__main__":
    main()
