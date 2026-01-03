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
# Configuration
# =============================================================================

APP_TITLE = "Universal Media Studio"
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent
# Storage directory for persistence (prefer /workspace on RunPod)
STORAGE_DIR = Path("/workspace") if Path("/workspace").exists() else BASE_DIR

MODELS_DIR = STORAGE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
OUTPUT_DIR = STORAGE_DIR / "outputs"
TEMP_DIR = STORAGE_DIR / "temp"
BIN_DIR = STORAGE_DIR / "bin"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables BEFORE importing heavy libraries
os.environ["PYTHONPATH"] = f"{BASE_DIR}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
os.environ["NLTK_DATA"] = str(MODELS_DIR / "nltk")
os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# Ensure Gradio uses the persistent temp directory
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)

# Now import heavy libraries and torch (after env vars are set)
import torch
import gradio as gr
import whisperx
import nltk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clear_model_cache():
    """Clear all possible model cache directories to resolve checksum issues."""
    cache_dirs = [
        WHISPER_DIR,
        MODELS_DIR / "whisper", 
        MODELS_DIR / "alignment",
        Path.home() / ".cache" / "whisper",
        Path.home() / ".cache" / "huggingface", 
        Path.home() / ".cache" / "whisperx",
        Path.home() / ".cache" / "torch" / "hub",
        Path("/tmp/whisperx")
    ]
    
    logger.info("🧹 Clearing model cache directories...")
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
    
    logger.info("✅ Cache clearing completed")


# Check NVENC and AV1 support
HAS_NVENC = False
HAS_AV1_NVENC = False
try:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    HAS_NVENC = "h264_nvenc" in r.stdout
    HAS_AV1_NVENC = "av1_nvenc" in r.stdout
except:
    pass


def cleanup_old_outputs(max_age_hours=24):
    """Remove output files and directories older than max_age_hours with network-volume resilience."""
    if not OUTPUT_DIR.exists():
        return
    now = datetime.now().timestamp()
    # Buffer to avoid deleting files currently being written (especially on network volumes)
    write_buffer_seconds = 60 
    for item in OUTPUT_DIR.iterdir():
        try:
            mtime = item.stat().st_mtime
            # Check if file has been modified recently (to avoid deleting active writes)
            if (now - mtime) > (max_age_hours * 3600) and (now - mtime) > write_buffer_seconds:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
        except Exception as e:
            logger.debug(f"Failed to cleanup {item}: {e}")

def cleanup_temp_dir():
    """Remove old temp directories on startup."""
    if TEMP_DIR.exists():
        for d in TEMP_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            else:
                try:
                    d.unlink()
                except:
                    pass
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Run cleanups on import
cleanup_temp_dir()
cleanup_old_outputs()

def flush_vram():
    """Clear GPU cache, collect garbage, and clear IPC memory."""
    gc.collect()
    if torch.cuda.is_available():
        # Clear specific caches that faster-whisper/pyannote might leave behind
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            # Force synchronization to ensure all kernels are finished
            torch.cuda.synchronize()
        except:
            pass
    # Additional garbage collection for heavy model objects
    gc.collect()

def check_disk_space(required_gb=2.0, path=STORAGE_DIR):
    """Ensure there is enough disk space available."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        if free_gb < required_gb:
            raise gr.Error(f"Insufficient disk space! {free_gb:.2f}GB available, but {required_gb}GB required.")
    except FileNotFoundError:
        # If path doesn't exist yet, check parent
        if not path.exists():
            check_disk_space(required_gb, path.parent)

def safe_move(src: Path, dst: Path):
    """Move a file atomically even across filesystems."""
    if not src.exists():
        return
    try:
        # Try atomic rename first (only works on same filesystem)
        src.rename(dst)
    except OSError:
        # Fallback for cross-filesystem move
        shutil.copy2(src, dst)
        src.unlink()


# =============================================================================
# Media Presets - Returns (batch_size,)
# =============================================================================

PRESETS = {
    "Standard": (16,),
    "High Accuracy": (8,),
    "Fast": (32,),
}


# =============================================================================
# Backend Functions
# =============================================================================

def get_path(input_val):
    """Robustly extract path from various Gradio component return types."""
    if input_val is None:
        return None
    if isinstance(input_val, dict):
        return input_val.get("path") or input_val.get("name")
    if isinstance(input_val, (list, tuple)) and len(input_val) > 0:
        return get_path(input_val[0])
    if hasattr(input_val, "name"):
        return input_val.name
    return str(input_val)


def generate_subtitles(
    audio_file,
    model_size,
    batch_size,
    language,
    hf_token,
    enable_diarization,
    *args,
    progress=gr.Progress()
) -> Tuple[str, Optional[str], Optional[str]]:
    """Generate subtitles using WhisperX with full feature utilization."""
    flush_vram()
    check_disk_space(1.0)
    
    audio_file = get_path(audio_file)
    if not audio_file:
        raise gr.Error("Please upload an audio or video file.")
    
    if not Path(audio_file).exists():
        raise gr.Error(f"File not found: {audio_file}")
    
    try:
        gr.Info("Starting transcription process...")
        progress(0.1, desc="Loading WhisperX...")
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        
        progress(0.2, desc="Loading audio...")
        audio = whisperx.load_audio(audio_file)
        
        progress(0.4, desc="Transcribing...")
        
        def _load_model():
            return whisperx.load_model(
                model_size, DEVICE,
                compute_type=compute_type,
                download_root=str(WHISPER_DIR)
            )

        try:
            model = _load_model()
        except Exception as e:
            err_msg = str(e).lower()
            if "sha256" in err_msg or "checksum" in err_msg:
                logger.warning(f"Checksum mismatch for {model_size}. Clearing all model caches and retrying...")
                clear_model_cache()
                
                # Retry with force reload
                try:
                    model = _load_model()
                    logger.info("✅ Model loaded successfully after cache clear")
                except Exception as retry_e:
                    if "sha256" in str(retry_e).lower() or "checksum" in str(retry_e).lower():
                        # Second retry with different approach
                        logger.warning("Second checksum failure, trying with fresh download...")
                        import time
                        time.sleep(2)  # Brief pause
                        model = _load_model()
                    else:
                        raise retry_e
            else:
                raise e

        result = model.transcribe(
            audio,
            batch_size=batch_size,
            language=language if language != "auto" else None
        )
        detected_lang = result["language"]
        
        # Explicitly delete model to free VRAM
        del model
        flush_vram()

        progress(0.6, desc="Aligning timestamps...")
        
        def _load_align():
            return whisperx.load_align_model(
                language_code=detected_lang, 
                device=DEVICE,
                model_dir=str(MODELS_DIR / "alignment")
            )
            
        try:
            model_a, metadata = _load_align()
        except Exception as e:
            err_msg = str(e).lower()
            if "sha256" in err_msg or "checksum" in err_msg:
                logger.warning(f"Checksum mismatch for alignment model. Clearing all model caches and retrying...")
                clear_model_cache()
                
                # Retry with force reload
                try:
                    model_a, metadata = _load_align()
                    logger.info("✅ Alignment model loaded successfully after cache clear")
                except Exception as retry_e:
                    if "sha256" in str(retry_e).lower() or "checksum" in str(retry_e).lower():
                        # Second retry with different approach
                        logger.warning("Second alignment checksum failure, trying with fresh download...")
                        import time
                        time.sleep(2)  # Brief pause
                        model_a, metadata = _load_align()
                    else:
                        raise retry_e
            else:
                raise e

        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
        
        del model_a
        flush_vram()

        # Speaker diarization
        if enable_diarization:
            if not hf_token:
                raise gr.Error(
                    "Speaker diarization requires a HuggingFace token.\n"
                    "1. Get token: huggingface.co/settings/tokens\n"
                    "2. Accept: huggingface.co/pyannote/segmentation-3.0\n"
                    "3. Accept: huggingface.co/pyannote/speaker-diarization-3.1"
                )
            progress(0.8, desc="Diarizing speakers...")
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
                flush_vram()
            except Exception as e:
                raise gr.Error(f"Diarization failed: {e}\nCheck your HuggingFace token and model agreements.")
        
        # Reclaim memory from audio array
        if 'audio' in locals():
            del audio
        flush_vram()
        
        progress(0.9, desc="Writing files...")
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = Path(audio_file).stem
        srt_path = OUTPUT_DIR / f"{name}_{ts}.srt"
        ass_path = OUTPUT_DIR / f"{name}_{ts}.ass"
        
        # Build transcript and SRT
        lines, srt = [], []
        for i, seg in enumerate(result["segments"], 1):
            text = seg["text"].strip()
            speaker = seg.get("speaker", "")
            lines.append(f"[{speaker}] {text}" if speaker else text)
            
            s, e = seg["start"], seg["end"]
            st = f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d},{int((s%1)*1000):03d}"
            et = f"{int(e//3600):02d}:{int((e%3600)//60):02d}:{int(e%60):02d},{int((e%1)*1000):03d}"
            srt.append(f"{i}\n{st} --> {et}\n{text}\n")
        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt))
        
        # ASS file
        ass_header = f"""[Script Info]
Title: Universal Media Studio
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1
"""
        events = []
        for seg in result["segments"]:
            s, e = seg["start"], seg["end"]
            st = f"{int(s//3600)}:{int((s%3600)//60):02d}:{(s%60):05.2f}"
            et = f"{int(e//3600)}:{int((e%3600)//60):02d}:{(e%60):05.2f}"
            events.append(f"Dialogue: 0,{st},{et},Default,,0,0,0,,{seg['text'].strip()}")
        
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(events))
        
        progress(1.0, desc="Complete!")
        return "\n".join(lines), str(srt_path), str(ass_path)
        
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")
    finally:
        flush_vram()


def burn_subtitles(video, subs, font_size, *args, progress=gr.Progress()) -> str:
    """Burn subtitles into video."""
    flush_vram()
    check_disk_space(2.0)

    video = get_path(video)
    subs = get_path(subs)
    
    if not video:
        raise gr.Error("Please upload a video file.")
    if not subs:
        raise gr.Error("Please upload a subtitle file.")
    
    sub_path = subs
    
    if not Path(sub_path).exists():
        raise gr.Error(f"Subtitle file not found: {sub_path}")
    
    try:
        gr.Info("Burning subtitles...")
        progress(0.2, desc="Burning subtitles...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"{Path(video).stem}_subtitled_{ts}.mp4"
        
        # Hardened escaping for FFmpeg filter paths (internal FFmpeg escaping)
        # For the subtitles/ass filters, the filename is a string within a filtergraph.
        # 1. Backslashes must be doubled
        # 2. Single quotes must be escaped as '\''
        # 3. Colons must be escaped if they are interpreted as separators
        
        # Use shlex.quote for the overall command list, 
        # but the internal filter string needs specific FFmpeg escaping.
        def ffmpeg_escape_path(path_str):
            return path_str.replace("\\", "/").replace("'", "'\\''").replace(":", "\\:")

        escaped_path = ffmpeg_escape_path(str(sub_path))
        
        # Use 'ass' filter for both .ass and .ssa files
        if sub_path.lower().endswith((".ass", ".ssa")):
            vf = f"ass='{escaped_path}'"
        else:
            # force_style uses its own comma-separated key=value format
            vf = f"subtitles='{escaped_path}':force_style='FontSize={font_size}'"
        
        cmd = ["ffmpeg", "-y", "-i", video, "-vf", vf]
        
        if HAS_NVENC:
            # Optimal settings for Ada/Ampere: p7 (highest quality), 10-bit if possible
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "20"])
        else:
            cmd.extend(["-c:v", "libx264", "-crf", "18"])
        cmd.extend(["-c:a", "copy", "-movflags", "+faststart", str(out)])
        
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200)
        
        progress(1.0, desc="Complete!")
        return str(out)
    except Exception as e:
        raise gr.Error(f"FFmpeg failed: {str(e)}")
    finally:
        flush_vram()


def convert_video(video, fmt, vcodec, acodec, crf, *args, progress=gr.Progress()) -> Optional[str]:
    """Convert video format."""
    flush_vram()
    check_disk_space(2.0)

    video = get_path(video)
    if not video:
        raise gr.Error("Please upload a video file.")
    
    if not Path(video).exists():
        raise gr.Error(f"Video file not found: {video}")
    
    try:
        gr.Info("Converting video...")
        progress(0.2, desc="Converting...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"{Path(video).stem}_converted_{ts}.{fmt}"
        
        cmd = ["ffmpeg", "-y", "-i", video]
        
        vc_map = {
            "H.264": ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"],
            "H.265": ["-c:v", "libx265", "-crf", str(crf), "-pix_fmt", "yuv420p"],
            "H.264 (NVENC)": ["-c:v", "h264_nvenc", "-cq", str(crf), "-pix_fmt", "yuv420p"],
            "H.265 (NVENC)": ["-c:v", "hevc_nvenc", "-cq", str(crf), "-pix_fmt", "yuv420p"],
            "VP9": ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"],
            "Copy": ["-c:v", "copy"]
        }
        cmd.extend(vc_map.get(vcodec, ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]))
        
        ac_map = {"AAC": ["-c:a", "aac", "-b:a", "192k"], "Opus": ["-c:a", "libopus", "-b:a", "128k"], "Copy": ["-c:a", "copy"]}
        cmd.extend(ac_map.get(acodec, ["-c:a", "aac"]))
        cmd.extend(["-movflags", "+faststart", str(out)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            raise gr.Error(f"FFmpeg conversion failed: {result.stderr}")
        
        progress(1.0, desc="Complete!")
        return str(out)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Video conversion failed: {str(e)}")
    finally:
        flush_vram()


def extract_audio(video, fmt, *args, progress=gr.Progress()) -> Optional[str]:
    """Extract audio from video."""
    flush_vram()
    check_disk_space(0.5)

    video = get_path(video)
    if not video:
        raise gr.Error("Please upload a video file.")
    
    if not Path(video).exists():
        raise gr.Error(f"Video file not found: {video}")
    
    try:
        gr.Info("Extracting audio...")
        progress(0.2, desc="Extracting audio...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"{Path(video).stem}_audio_{ts}.{fmt}"
        
        codec_map = {
            "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
            "wav": ["-c:a", "pcm_s16le"],
            "flac": ["-c:a", "flac"],
            "aac": ["-c:a", "aac", "-b:a", "192k"],
            "opus": ["-c:a", "libopus", "-b:a", "128k"]
        }
        
        cmd = ["ffmpeg", "-y", "-i", video, "-vn"] + codec_map.get(fmt, []) + [str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise gr.Error(f"FFmpeg audio extraction failed: {result.stderr}")
        
        progress(1.0, desc="Complete!")
        return str(out)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Audio extraction failed: {str(e)}")
    finally:
        flush_vram()


# =============================================================================
# Gradio UI - Clean Dashboard Layout
# =============================================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}
footer { display: none !important; }
.primary-btn { min-height: 45px !important; }
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 500;
}
.status-success { background: #dcfce7; color: #166534; }
.status-warning { background: #fef3c7; color: #92400e; }
.output-panel { border-left: 3px solid #3b82f6; padding-left: 1rem; }
"""

def create_ui():
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
        ),
        css=CUSTOM_CSS
    ) as app:
        
        # Header with status indicators
        gpu_status = f"<span class='status-badge status-success'>✓ {DEVICE.upper()}</span>" if DEVICE == "cuda" else "<span class='status-badge status-warning'>CPU Mode</span>"
        nvenc_status = "<span class='status-badge status-success'>✓ NVENC</span>" if HAS_NVENC else "<span class='status-badge status-warning'>Software Encoding</span>"
        
        gr.Markdown(f"""
        # 🎬 {APP_TITLE}
        **Professional AI-Powered Media Processing**
        
        {gpu_status} {nvenc_status}
        """)
        
        # =====================================================================
        # Global Preset Bar
        # =====================================================================
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()), value="Live Action",
                label="🎯 Media Preset",
                info="Auto-configures optimal settings for your content type",
                scale=2,
                interactive=True
            )
            with gr.Column(scale=4):
                gr.Markdown(
                    "<div style='padding: 10px; background: #f1f5f9; border-radius: 8px; font-size: 0.9em;'>"
                    "💡 <b>Tip:</b> Select a preset to automatically configure AI models for best results."
                    "</div>"
                )
        
        # =====================================================================
        # Tab 1: AI Subtitles
        # =====================================================================
        with gr.Tabs():
            with gr.Tab("🎤 AI Subtitles", id="subtitles"):
                gr.Markdown(
                    "### Generate accurate, time-aligned subtitles\n"
                    "*Powered by WhisperX with word-level alignment and optional speaker diarization*"
                )
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        sub_input = gr.Audio(
                            label="📁 Upload Audio/Video",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            sub_model = gr.Dropdown(
                                choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                                value="large-v3",
                                label="Whisper Model",
                                info="large-v3 is most accurate, tiny is fastest"
                            )
                            sub_batch = gr.Slider(
                                1, 32, value=16, step=1,
                                label="Batch Size",
                                info="Higher = faster but more VRAM"
                            )
                            sub_lang = gr.Dropdown(
                                choices=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "vi"],
                                value="auto",
                                label="Language",
                                info="Auto-detect or specify for better accuracy"
                            )
                            gr.Markdown("---")
                            sub_diarize = gr.Checkbox(
                                label="🎙️ Speaker Diarization",
                                value=False,
                                info="Identify different speakers (requires HF token)"
                            )
                            sub_token = gr.Textbox(
                                label="HuggingFace Token",
                                type="password",
                                placeholder="hf_...",
                                info="Required for diarization - get from huggingface.co/settings/tokens"
                            )
                        
                        sub_btn = gr.Button("🚀 Generate Subtitles", variant="primary", size="lg", elem_classes=["primary-btn"])
                    
                    with gr.Column(scale=1, elem_classes=["output-panel"]):
                        gr.Markdown("#### 📤 Output")
                        sub_transcript = gr.Textbox(
                            label="📝 Transcript",
                            lines=10,
                            interactive=True,
                            show_copy_button=True,
                            placeholder="Generated transcript will appear here..."
                        )
                        with gr.Row():
                            sub_srt = gr.File(label="📄 SRT File", file_count="single")
                            sub_ass = gr.File(label="📄 ASS File", file_count="single")

            # =====================================================================
            # Tab 2: Toolbox
            # =====================================================================
            with gr.Tab("🛠️ Toolbox", id="toolbox"):
                with gr.Tabs():

                    # Tab 2.1: Burn Subtitles
                    with gr.Tab("🔥 Burn Subtitles"):
                        gr.Markdown("### Hardcode subtitles into video permanently")
                        with gr.Row():
                            with gr.Column():
                                burn_vid = gr.Video(label="Video Source")
                                burn_sub = gr.File(label="Subtitle File (.srt, .ass, .vtt)", type="filepath")
                                burn_font = gr.Slider(10, 100, value=24, step=2, label="Font Size")
                                burn_btn = gr.Button("🔥 Burn Subtitles", variant="primary")
                            with gr.Column(elem_classes=["output-panel"]):
                                burn_out = gr.Video(label="Output Video")

                    # Tab 3.2: Convert
                    with gr.Tab("🔄 Convert"):
                        gr.Markdown("### Transcode video formats")
                        with gr.Row():
                            with gr.Column():
                                conv_in = gr.Video(label="Input Video")
                                with gr.Row():
                                    conv_fmt = gr.Dropdown(["mp4", "mkv", "mov", "webm"], value="mp4", label="Format")
                                    conv_vc = gr.Dropdown(["H.264", "H.265", "H.264 (NVENC)", "H.265 (NVENC)", "VP9", "Copy"], value="H.264 (NVENC)" if HAS_NVENC else "H.264", label="Video Codec")
                                with gr.Row():
                                    conv_ac = gr.Dropdown(["AAC", "Opus", "Copy"], value="AAC", label="Audio Codec")
                                    conv_crf = gr.Slider(0, 51, value=23, step=1, label="Quality (CRF)", info="Lower is better, 18-28 is standard")
                                conv_btn = gr.Button("🔄 Convert", variant="primary")
                            with gr.Column(elem_classes=["output-panel"]):
                                conv_out = gr.Video(label="Converted Video")

                    # Tab 2.3: Extract Audio
                    with gr.Tab("🎵 Extract Audio"):
                        gr.Markdown("### Extract audio track from video")
                        with gr.Row():
                            with gr.Column():
                                ext_in = gr.Video(label="Input Video")
                                ext_fmt = gr.Dropdown(["mp3", "wav", "flac", "aac", "opus"], value="mp3", label="Output Format")
                                ext_btn = gr.Button("🎵 Extract", variant="primary")
                            with gr.Column(elem_classes=["output-panel"]):
                                ext_out = gr.Audio(label="Extracted Audio", type="filepath")

                    # Tab 2.4: System Maintenance
                    with gr.Tab("⚙️ System"):
                        gr.Markdown("### System Maintenance")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown(
                                    "#### 🧹 Model Cache Management\n"
                                    "If you encounter 'Checksum Mismatch' errors or transcription fails unexpectedly, "
                                    "try clearing the model cache. This will force the app to re-download the required AI models."
                                )
                                clear_cache_btn = gr.Button("🗑️ Clear All Model Caches", variant="stop")
                                cache_status = gr.Markdown("*Status: Ready*")
                            
                            with gr.Column(elem_classes=["output-panel"]):
                                gr.Markdown(
                                    "#### 📊 Storage Info\n"
                                    "Monitor disk usage on your instance."
                                )
                                storage_info_btn = gr.Button("🔄 Refresh Storage Info")
                                storage_display = gr.Code(label="Disk Usage", language="bash")

                # Connect UI actions
                def handle_clear_cache():
                    try:
                        clear_model_cache()
                        return "✅ Cache cleared successfully! Models will be re-downloaded on next use."
                    except Exception as e:
                        return f"❌ Error clearing cache: {str(e)}"

                def get_storage_info():
                    try:
                        result = subprocess.run(["df", "-h", str(STORAGE_DIR)], capture_output=True, text=True)
                        return result.stdout
                    except Exception as e:
                        return f"Error: {str(e)}"

                clear_cache_btn.click(fn=handle_clear_cache, outputs=[cache_status])
                storage_info_btn.click(fn=get_storage_info, outputs=[storage_display])

                sub_btn.click(
                    fn=generate_subtitles,
                    inputs=[sub_input, sub_model, sub_batch, sub_lang, sub_token, sub_diarize],
                    outputs=[sub_transcript, sub_srt, sub_ass]
                )
                
                # Preset connection
                preset_dropdown.change(
                    fn=on_preset_change,
                    inputs=[preset_dropdown],
                    outputs=[sub_batch]
                )
                
                # Toolbox tools
                burn_btn.click(burn_subtitles, [burn_vid, burn_sub, burn_font], burn_out)
                conv_btn.click(convert_video, [conv_in, conv_fmt, conv_vc, conv_ac, conv_crf], conv_out)
                ext_btn.click(extract_audio, [ext_in, ext_fmt], ext_out)
        
        # Footer
        gr.Markdown(
            "---\n"
            "<center>"
            "<small>🎬 <b>Universal Media Studio</b> | "
            "WhisperX • FFmpeg Toolbox</small>\n"
            "<small style='color: #64748b;'>Built for RunPod GPU Instances</small>"
            "</center>"
        )
    
    return app


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("="*60)
    logger.info(f"🚀 {APP_TITLE} Starting...")
    logger.info("="*60)
    logger.info(f"📍 Base Dir:    {BASE_DIR}")
    logger.info(f"📍 Models Dir:  {MODELS_DIR}")
    logger.info(f"📍 Output Dir:  {OUTPUT_DIR}")
    logger.info(f"📍 Temp Dir:    {TEMP_DIR}")
    logger.info(f"📍 Device:      {DEVICE}")
    logger.info(f"📍 NVENC:       {HAS_NVENC}")
    logger.info(f"📍 PYTHONPATH:  {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info("="*60)
    
    app = create_ui()
    app.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        allowed_paths=["/app", "/workspace", "/tmp"],
        root_path=os.environ.get("GRADIO_ROOT_PATH", None)
    )
