# =============================================================================
# Universal Media Studio - Model Downloader
# Downloads all required models at Docker build time for instant RunPod startup.
# =============================================================================

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
logger = logging.getLogger("Downloader")

# FIX: Global monkey patch for BasicSR compatibility with torchvision >= 0.16
# BasicSR expects torchvision.transforms.functional_tensor which was removed
try:
    import torchvision
    import torchvision.transforms.functional as F
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

# Paths - robust to local vs Docker
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent
# Storage directory for persistence (prefer /workspace on RunPod)
STORAGE_DIR = Path("/workspace") if Path("/workspace").exists() else BASE_DIR

MODELS_DIR = STORAGE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
HF_HOME = MODELS_DIR / "huggingface"
NLTK_HOME = MODELS_DIR / "nltk"
TORCH_HOME = MODELS_DIR / "torch"
CODEFORMER_DIR = MODELS_DIR / "CodeFormer"
BIN_DIR = STORAGE_DIR / "bin"

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
        Path.home() / ".cache" / "torch" / "hub"
    ]
    
    logger.info("🧹 Clearing all model cache directories...")
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            logger.info(f"  Removing: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Recreate essential directories
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "alignment").mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    NLTK_HOME.mkdir(parents=True, exist_ok=True)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Cache clearing completed")


# Ensure directories exist
os.environ["FACEXLIB_HOME"] = str(MODELS_DIR / "facexlib")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Ensure directories exist
WHISPER_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)
NLTK_HOME.mkdir(parents=True, exist_ok=True)
TORCH_HOME.mkdir(parents=True, exist_ok=True)
Path(os.environ["FACEXLIB_HOME"]).mkdir(parents=True, exist_ok=True)


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


def download_vad_model():
    """Download Silero VAD model used by WhisperX with checksum resilience."""
    logger.info("=" * 60)
    logger.info("📥 Downloading VAD model...")
    logger.info("=" * 60)
    
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
            if "checksum" in str(e).lower():
                logger.warning("VAD checksum mismatch. Clearing all caches and retrying with force_reload=True...")
                clear_all_model_caches()
                
                # First retry with force reload
                try:
                    torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=True,
                        onnx=False
                    )
                    logger.info("✅ VAD model loaded successfully after cache clear")
                except Exception as retry_e:
                    if "checksum" in str(retry_e).lower():
                        # Second retry with pause
                        logger.warning("Second VAD checksum failure, trying with fresh download...")
                        import time
                        time.sleep(3)
                        torch.hub.load(
                            repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=True,
                            onnx=False
                        )
                    else:
                        raise retry_e
            else:
                raise e
                
        logger.info("✅ Silero VAD model downloaded!")
        
        # WhisperX specific VAD segmentation model
        # Fixes 301 Redirect issue with old URL in whisperx < 3.1.2
        logger.info("  Downloading WhisperX VAD segmentation model...")

        # This URL is the target of the redirect from the hardcoded EU URL
        # We manually download it so whisperx doesn't try to use the broken URL
        url = "https://whisperx.s3.us-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"

        # Target location expected by WhisperX
        # It looks in torch.hub._get_torch_home() -> which maps to os.environ["TORCH_HOME"]
        target_path = TORCH_HOME / "whisperx-vad-segmentation.bin"

        if not target_path.exists():
            # Use curl -L to handle any further redirects robustly
            # We use a user-agent to avoid 403s on some S3 setups
            subprocess.run([
                "curl", "-L",
                "-A", "Mozilla/5.0",
                "-o", str(target_path),
                url
            ], check=True)
            logger.info("  ✅ WhisperX VAD segmentation model downloaded!")
        else:
            logger.info("  ✓ WhisperX VAD segmentation model already exists")

    except Exception as e:
        logger.error(f"⚠️ VAD model download error: {e}")


def download_codeformer_models():
    """Download CodeFormer pretrained models."""
    logger.info("=" * 60)
    logger.info("📥 Downloading CodeFormer models...")
    logger.info("=" * 60)
    
    if not CODEFORMER_DIR.exists():
        logger.warning("⚠️ CodeFormer directory not found, skipping...")
        return
    
    try:
        # Run CodeFormer's download script
        download_script = CODEFORMER_DIR / "scripts" / "download_pretrained_models.py"
        
        if download_script.exists():
            subprocess.run(
                [sys.executable, str(download_script), "facelib"],
                cwd=str(CODEFORMER_DIR),
                check=True
            )
            subprocess.run(
                [sys.executable, str(download_script), "CodeFormer"],
                cwd=str(CODEFORMER_DIR),
                check=True
            )
            logger.info("✅ CodeFormer models downloaded!")
        else:
            logger.warning("⚠️ CodeFormer download script not found")
            
    except Exception as e:
        logger.error(f"⚠️ CodeFormer download error: {e}")


def download_demucs_models():
    """Pre-download Demucs models offered in the UI."""
    logger.info("=" * 60)
    logger.info("📥 Downloading Demucs models...")
    logger.info("=" * 60)
    
    try:
        from demucs.pretrained import get_model
        
        # Models offered in app.py UI
        models = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"]
        for model_name in models:
            logger.info(f"  Downloading: {model_name}...")
            model = get_model(model_name)
            del model
            logger.info(f"  ✅ {model_name} ready")
            
    except Exception as e:
        logger.error(f"⚠️ Demucs download error: {e}")


def download_deepfilternet_models():
    """Pre-download DeepFilterNet models."""
    logger.info("=" * 60)
    logger.info("📥 Downloading DeepFilterNet models...")
    logger.info("=" * 60)
    
    try:
        from df.enhance import init_df
        
        # Initialize to trigger download
        model, df_state, _ = init_df()
        del model, df_state
        logger.info("✅ DeepFilterNet models downloaded!")
        
    except Exception as e:
        logger.error(f"⚠️ DeepFilterNet download error: {e}")


def verify_realesrgan_binary():
    """Verify realesrgan-ncnn-vulkan binary is ready."""
    logger.info("=" * 60)
    logger.info("🔍 Verifying Real-ESRGAN binary...")
    logger.info("=" * 60)
    
    binary_path = BIN_DIR / "realesrgan-ncnn-vulkan"
    
    if binary_path.exists():
        # Make executable
        os.chmod(binary_path, 0o755)
        logger.info(f"✅ Real-ESRGAN binary found at: {binary_path}")
        
        # Verify models exist
        models_dir = BIN_DIR / "models"
        if models_dir.exists():
            models = list(models_dir.glob("*.bin"))
            logger.info(f"  Found {len(models)} model files")
        else:
            logger.warning("  ⚠️ Models directory not found")
    else:
        logger.warning(f"⚠️ Real-ESRGAN binary not found at: {binary_path}")


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
    download_demucs_models()
    download_deepfilternet_models()
    download_codeformer_models()
    verify_realesrgan_binary()
    
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
