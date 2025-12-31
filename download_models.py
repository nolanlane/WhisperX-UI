# =============================================================================
# Universal Media Studio - Model Downloader
# Downloads all required models at Docker build time for instant RunPod startup.
# =============================================================================

import os
import sys
import subprocess
from pathlib import Path

# FIX: Monkey patch for BasicSR compatibility with torchvision >= 0.16
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
MODELS_DIR = BASE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
HF_HOME = MODELS_DIR / "huggingface"
NLTK_HOME = MODELS_DIR / "nltk"
TORCH_HOME = MODELS_DIR / "torch"
CODEFORMER_DIR = MODELS_DIR / "CodeFormer"
BIN_DIR = BASE_DIR / "bin"

# Set HuggingFace, NLTK, and Torch cache locations
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME)
os.environ["NLTK_DATA"] = str(NLTK_HOME)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Ensure directories exist
WHISPER_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)
NLTK_HOME.mkdir(parents=True, exist_ok=True)
TORCH_HOME.mkdir(parents=True, exist_ok=True)


def download_whisper_model():
    """Download WhisperX large-v3 model."""
    print("\n" + "=" * 60)
    print("📥 Downloading Whisper large-v3 model...")
    print("=" * 60)
    
    try:
        import whisperx
        import nltk
        
        # WhisperX uses nltk for sentence segmentation
        print(f"  Downloading NLTK data to {os.environ['NLTK_DATA']}...")
        nltk.download('punkt', download_dir=os.environ["NLTK_DATA"])
        nltk.download('punkt_tab', download_dir=os.environ["NLTK_DATA"])
        
        # Load model to trigger download (uses faster-whisper under the hood)
        # CRITICAL: Use int8 on CPU, NOT float16 (float16 requires GPU)
        model = whisperx.load_model(
            "large-v3",
            device="cpu",
            compute_type="int8",
            download_root=str(WHISPER_DIR)
        )
        del model
        print("✅ Whisper large-v3 downloaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Whisper download error (may still work at runtime): {e}")


def download_alignment_models():
    """Download WhisperX alignment models for common languages."""
    print("\n" + "=" * 60)
    print("📥 Downloading alignment models...")
    print("=" * 60)
    
    try:
        import whisperx
        import torch
        
        # Download common alignment models to match UI choices
        languages = ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "vi"]
        for lang in languages:
            print(f"  Downloading alignment model for: {lang}")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=lang,
                    device="cpu",
                    model_dir=str(MODELS_DIR / "alignment")
                )
                del model_a
                print(f"  ✅ {lang} alignment model ready")
            except Exception as e:
                print(f"  ⚠️ {lang} alignment model: {e}")
        
    except Exception as e:
        print(f"⚠️ Alignment model download error: {e}")


def download_vad_model():
    """Download Silero VAD model used by WhisperX."""
    print("\n" + "=" * 60)
    print("📥 Downloading VAD model...")
    print("=" * 60)
    
    try:
        import torch
        
        # Silero VAD
        torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        print("✅ Silero VAD model downloaded!")
        
    except Exception as e:
        print(f"⚠️ VAD model download error: {e}")


def download_codeformer_models():
    """Download CodeFormer pretrained models."""
    print("\n" + "=" * 60)
    print("📥 Downloading CodeFormer models...")
    print("=" * 60)
    
    if not CODEFORMER_DIR.exists():
        print("⚠️ CodeFormer directory not found, skipping...")
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
            print("✅ CodeFormer models downloaded!")
        else:
            print("⚠️ CodeFormer download script not found")
            
    except Exception as e:
        print(f"⚠️ CodeFormer download error: {e}")


def download_demucs_models():
    """Pre-download Demucs models."""
    print("\n" + "=" * 60)
    print("📥 Downloading Demucs models...")
    print("=" * 60)
    
    try:
        # Import triggers model download
        from demucs.pretrained import get_model
        
        # Download htdemucs (default model)
        model = get_model("htdemucs")
        del model
        print("✅ Demucs htdemucs model downloaded!")
        
    except Exception as e:
        print(f"⚠️ Demucs download error: {e}")


def download_deepfilternet_models():
    """Pre-download DeepFilterNet models."""
    print("\n" + "=" * 60)
    print("📥 Downloading DeepFilterNet models...")
    print("=" * 60)
    
    try:
        from df.enhance import init_df
        
        # Initialize to trigger download
        model, df_state, _ = init_df()
        del model, df_state
        print("✅ DeepFilterNet models downloaded!")
        
    except Exception as e:
        print(f"⚠️ DeepFilterNet download error: {e}")


def verify_realesrgan_binary():
    """Verify realesrgan-ncnn-vulkan binary is ready."""
    print("\n" + "=" * 60)
    print("🔍 Verifying Real-ESRGAN binary...")
    print("=" * 60)
    
    binary_path = BIN_DIR / "realesrgan-ncnn-vulkan"
    
    if binary_path.exists():
        # Make executable
        os.chmod(binary_path, 0o755)
        print(f"✅ Real-ESRGAN binary found at: {binary_path}")
        
        # Verify models exist
        models_dir = BIN_DIR / "models"
        if models_dir.exists():
            models = list(models_dir.glob("*.bin"))
            print(f"  Found {len(models)} model files")
        else:
            print("  ⚠️ Models directory not found")
    else:
        print(f"⚠️ Real-ESRGAN binary not found at: {binary_path}")


def download_gfpgan_models():
    """Download GFPGAN models for face enhancement."""
    print("\n" + "=" * 60)
    print("📥 Downloading GFPGAN models...")
    print("=" * 60)
    
    try:
        from gfpgan import GFPGANer
        import urllib.request
        
        # Download GFPGANv1.4 model
        model_path = MODELS_DIR / "gfpgan" / "GFPGANv1.4.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not model_path.exists():
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
            print(f"  Downloading from: {url}")
            urllib.request.urlretrieve(url, str(model_path))
        
        print("✅ GFPGAN models downloaded!")
        
    except Exception as e:
        print(f"⚠️ GFPGAN download error: {e}")


def main():
    """Main download orchestrator."""
    print("\n" + "=" * 60)
    print("🚀 Universal Media Studio - Model Downloader")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}")
    print(f"HuggingFace cache: {HF_HOME}")
    
    # Run all downloads
    download_whisper_model()
    download_alignment_models()
    download_vad_model()
    download_demucs_models()
    download_deepfilternet_models()
    download_codeformer_models()
    download_gfpgan_models()
    verify_realesrgan_binary()
    
    print("\n" + "=" * 60)
    print("✅ Model download complete!")
    print("=" * 60)
    
    # Print cache sizes
    import shutil
    
    def get_dir_size(path):
        if path.exists():
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total / (1024 ** 3)  # GB
        return 0
    
    print(f"\n📊 Cache sizes:")
    print(f"  Whisper models: {get_dir_size(WHISPER_DIR):.2f} GB")
    print(f"  HuggingFace cache: {get_dir_size(HF_HOME):.2f} GB")
    print(f"  Total: {get_dir_size(MODELS_DIR):.2f} GB")


if __name__ == "__main__":
    main()
