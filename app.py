#!/usr/bin/env python3
"""
Universal Media Studio - Production-Ready Gradio Application
Clean, dashboard-style interface for AI-powered media processing.
"""

import gc
import os
import sys
import shutil
import logging
import subprocess
import shlex
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

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
logger = logging.getLogger("UniversalMediaStudio")

# =============================================================================
# Configuration & Storage Detection
# =============================================================================

APP_TITLE = "Universal Media Studio"
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent

def find_storage_dir():
    """Robustly find a writable storage directory."""
    candidates = [
        Path("/workspace"), # RunPod standard
        Path("/app/data"),
        Path("/tmp/ums_data")
    ]

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test writability
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            logger.info(f"✅ Using storage directory: {path}")
            return path
        except Exception as e:
            logger.debug(f"❌ Cannot write to {path}: {e}")
            continue

    # Fallback to temp dir
    fallback = Path("/tmp/ums_data")
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning(f"⚠️ Falling back to temporary storage: {fallback}")
    return fallback

STORAGE_DIR = find_storage_dir()

MODELS_DIR = STORAGE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
HF_HOME = MODELS_DIR / "huggingface"
NLTK_HOME = MODELS_DIR / "nltk"
TORCH_HOME = MODELS_DIR / "torch"
OUTPUT_DIR = STORAGE_DIR / "outputs"
TEMP_DIR = STORAGE_DIR / "temp"

# Create directories
for d in [MODELS_DIR, WHISPER_DIR, HF_HOME, NLTK_HOME, TORCH_HOME, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Set environment variables BEFORE importing heavy libraries
os.environ["PYTHONPATH"] = f"{BASE_DIR}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["NLTK_DATA"] = str(NLTK_HOME)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)

# Now import heavy libraries and torch (after env vars are set)
import torch
import gradio as gr
import nltk

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ensure_vad_model():
    """Ensure the specific VAD segmentation model required by WhisperX is present."""
    target_path = TORCH_HOME / "whisperx-vad-segmentation.bin"
    expected_hash = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"
    
    if target_path.exists():
        if calculate_sha256(target_path) == expected_hash:
            return # Already good
        else:
            logger.warning("VAD model checksum mismatch. Re-downloading...")
            target_path.unlink()

    logger.info("Downloading VAD model...")
    urls = [
        "https://whisperx.s3.us-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin",
        "https://github.com/m-bain/whisperX/raw/main/whisperx/assets/pytorch_model.bin"
    ]

    for url in urls:
        try:
            subprocess.run(
                ["curl", "-L", "-f", "-o", str(target_path), url],
                check=True, timeout=300
            )
            if calculate_sha256(target_path) == expected_hash:
                logger.info("✅ VAD model downloaded successfully.")
                return
        except Exception as e:
            logger.warning(f"Failed to download VAD from {url}: {e}")

    logger.error("❌ Failed to download VAD model. WhisperX may fail.")

def clear_model_cache():
    """Clear model caches to resolve issues."""
    logger.info("🧹 Clearing model cache...")
    try:
        shutil.rmtree(WHISPER_DIR, ignore_errors=True)
        shutil.rmtree(MODELS_DIR / "alignment", ignore_errors=True)
        # Recreate
        WHISPER_DIR.mkdir(exist_ok=True)
        (MODELS_DIR / "alignment").mkdir(exist_ok=True)
        # Remove VAD
        (TORCH_HOME / "whisperx-vad-segmentation.bin").unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Cache clear warning: {e}")

def flush_vram():
    """Clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def get_path(input_val):
    """Robustly extract path from Gradio inputs."""
    if input_val is None: return None
    if isinstance(input_val, dict): return input_val.get("path") or input_val.get("name")
    if hasattr(input_val, "name"): return input_val.name
    return str(input_val)

# Check FFmpeg support
HAS_NVENC = False
try:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    HAS_NVENC = "h264_nvenc" in r.stdout
except:
    pass

# =============================================================================
# Core Logic
# =============================================================================

def generate_subtitles(
    audio_file, model_size, batch_size, language, hf_token, enable_diarization,
    progress=gr.Progress()
):
    """Generate subtitles with lazy model loading."""
    flush_vram()
    
    path = get_path(audio_file)
    if not path or not Path(path).exists():
        raise gr.Error("File not found.")
    
    try:
        import whisperx
        
        # 1. Ensure VAD model is present (Lazy Download)
        progress(0.05, desc="Checking models...")
        ensure_vad_model()
        
        # 2. Load Whisper Model
        progress(0.1, desc=f"Loading Whisper ({model_size})...")
        try:
            model = whisperx.load_model(
                model_size, DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=str(WHISPER_DIR)
            )
        except Exception as e:
            if "sha256" in str(e).lower():
                logger.warning("Checksum error. Clearing cache and retrying...")
                clear_model_cache()
                ensure_vad_model()
                model = whisperx.load_model(
                    model_size, DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=str(WHISPER_DIR)
                )
            else:
                raise e

        # 3. Transcribe
        progress(0.3, desc="Transcribing...")
        audio = whisperx.load_audio(path)
        result = model.transcribe(
            audio, batch_size=batch_size,
            language=language if language != "auto" else None
        )
        detected_lang = result["language"]
        del model
        flush_vram()
        
        # 4. Align
        progress(0.6, desc="Aligning...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang, 
                device=DEVICE,
                model_dir=str(MODELS_DIR / "alignment")
            )
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            del model_a
            flush_vram()
        except Exception as e:
            logger.warning(f"Alignment failed: {e}. Returning unaligned transcript.")
        
        # 5. Diarization
        if enable_diarization:
            if not hf_token:
                raise gr.Error("Diarization requires HuggingFace Token.")
            progress(0.8, desc="Diarizing...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            flush_vram()

        # 6. Output Generation
        progress(0.9, desc="Saving...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        srt_path = OUTPUT_DIR / f"{Path(path).stem}_{ts}.srt"
        ass_path = OUTPUT_DIR / f"{Path(path).stem}_{ts}.ass"
        
        # Write SRT
        srt_content = []
        transcript_text = []
        for i, seg in enumerate(result["segments"], 1):
            text = seg["text"].strip()
            speaker = seg.get("speaker", "")
            line = f"[{speaker}] {text}" if speaker else text
            transcript_text.append(line)
            
            s, e = seg["start"], seg["end"]
            st = f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d},{int((s%1)*1000):03d}"
            et = f"{int(e//3600):02d}:{int((e%3600)//60):02d}:{int(e%60):02d},{int((e%1)*1000):03d}"
            srt_content.append(f"{i}\n{st} --> {et}\n{text}\n")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        # Write ASS (Simplified)
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(f"[Script Info]\nTitle: AutoGen\n[Events]\nFormat: Start, End, Text\n")
            for seg in result["segments"]:
                s, e = seg["start"], seg["end"]
                st = f"{int(s//3600)}:{int((s%3600)//60):02d}:{(s%60):05.2f}"
                et = f"{int(e//3600)}:{int((e%3600)//60):02d}:{(e%60):05.2f}"
                f.write(f"Dialogue: {st},{et},{seg['text'].strip()}\n")

        return "\n".join(transcript_text), str(srt_path), str(ass_path)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise gr.Error(str(e))
    finally:
        flush_vram()

# =============================================================================
# Toolbox Functions (Simplified)
# =============================================================================

def run_ffmpeg(cmd, output_path):
    """Run ffmpeg command safely."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1200)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"FFmpeg Error: {e.stderr}")

def burn_subtitles(video, subs, font_size):
    video, subs = get_path(video), get_path(subs)
    if not video or not subs: raise gr.Error("Missing files.")
    
    out = OUTPUT_DIR / f"{Path(video).stem}_burned.mp4"
    # Escaping for filter complex
    sub_path_esc = str(subs).replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")
    vf = f"subtitles='{sub_path_esc}':force_style='FontSize={font_size}'"
    
    cmd = ["ffmpeg", "-y", "-i", video, "-vf", vf]
    if HAS_NVENC:
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "20"])
    else:
        cmd.extend(["-c:v", "libx264", "-crf", "23"])
    cmd.extend(["-c:a", "copy", str(out)])

    return run_ffmpeg(cmd, out)

def convert_video(video, fmt, vcodec, acodec, crf):
    video = get_path(video)
    if not video: raise gr.Error("No video.")
    
    out = OUTPUT_DIR / f"{Path(video).stem}_conv.{fmt}"
    cmd = ["ffmpeg", "-y", "-i", video]
    
    # Video Codec
    if "NVENC" in vcodec and HAS_NVENC:
        cmd.extend(["-c:v", "h264_nvenc" if "H.264" in vcodec else "hevc_nvenc", "-cq", str(crf)])
    elif "Copy" in vcodec:
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(["-c:v", "libx264", "-crf", str(crf)])
        
    # Audio Codec
    if "AAC" in acodec: cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    elif "Opus" in acodec: cmd.extend(["-c:a", "libopus", "-b:a", "128k"])
    else: cmd.extend(["-c:a", "copy"])

    cmd.append(str(out))
    return run_ffmpeg(cmd, out)

def extract_audio(video, fmt):
    video = get_path(video)
    if not video: raise gr.Error("No video.")
    out = OUTPUT_DIR / f"{Path(video).stem}.{fmt}"
    cmd = ["ffmpeg", "-y", "-i", video, "-vn", str(out)]
    return run_ffmpeg(cmd, out)

# =============================================================================
# UI Setup
# =============================================================================

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
footer { display: none !important; }
"""

def create_ui():
    with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS, theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# 🎬 {APP_TITLE}\n**GPU:** {DEVICE} | **NVENC:** {'Yes' if HAS_NVENC else 'No'}")
        
        with gr.Tabs():
            with gr.Tab("🎤 Subtitles"):
                with gr.Row():
                    with gr.Column():
                        inp_file = gr.File(label="Media File")
                        with gr.Accordion("Settings", open=False):
                            model_sz = gr.Dropdown(["tiny", "base", "small", "medium", "large-v2", "large-v3"], value="large-v3", label="Model")
                            batch_sz = gr.Slider(1, 64, 16, step=1, label="Batch Size")
                            lang = gr.Dropdown(["auto", "en", "es", "fr", "de", "ja", "zh"], value="auto", label="Language")
                            diarize = gr.Checkbox(label="Diarization")
                            token = gr.Textbox(label="HF Token (for Diarization)", type="password")
                        btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        out_txt = gr.Textbox(label="Transcript", lines=10)
                        out_srt = gr.File(label="SRT")
                        out_ass = gr.File(label="ASS")
                
                btn.click(generate_subtitles, [inp_file, model_sz, batch_sz, lang, token, diarize], [out_txt, out_srt, out_ass])

            with gr.Tab("🛠️ Toolbox"):
                with gr.Tab("Burn Subtitles"):
                    b_vid = gr.File(label="Video")
                    b_sub = gr.File(label="Subtitle")
                    b_font = gr.Slider(10, 50, 24, label="Font Size")
                    b_btn = gr.Button("Burn")
                    b_out = gr.Video()
                    b_btn.click(burn_subtitles, [b_vid, b_sub, b_font], b_out)
                
                with gr.Tab("Convert"):
                    c_vid = gr.File(label="Video")
                    c_fmt = gr.Dropdown(["mp4", "mkv", "webm"], value="mp4", label="Format")
                    c_vc = gr.Dropdown(["H.264", "H.264 (NVENC)", "Copy"], value="H.264 (NVENC)" if HAS_NVENC else "H.264", label="Video Codec")
                    c_ac = gr.Dropdown(["AAC", "Opus", "Copy"], value="AAC", label="Audio Codec")
                    c_crf = gr.Slider(0, 51, 23, label="Quality (CRF)")
                    c_btn = gr.Button("Convert")
                    c_out = gr.Video()
                    c_btn.click(convert_video, [c_vid, c_fmt, c_vc, c_ac, c_crf], c_out)

                with gr.Tab("Extract Audio"):
                    e_vid = gr.File(label="Video")
                    e_fmt = gr.Dropdown(["mp3", "wav", "flac"], value="mp3", label="Format")
                    e_btn = gr.Button("Extract")
                    e_out = gr.File()
                    e_btn.click(extract_audio, [e_vid, e_fmt], e_out)

    return app

if __name__ == "__main__":
    # Startup check
    logger.info(f"Starting {APP_TITLE} on {DEVICE}")
    app = create_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=["/"], # Allow all paths for container flexibility
        root_path=os.environ.get("GRADIO_ROOT_PATH", None)
    )
