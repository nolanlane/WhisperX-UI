#!/usr/bin/env python3
"""
Universal Media Studio - Production-Ready Gradio Application
Clean, dashboard-style interface for AI-powered media processing.
"""

import gc
import os
import sys
import json
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import torch
import gradio as gr

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

# FIX: Global monkey patch for BasicSR compatibility with torchvision >= 0.16
# BasicSR expects torchvision.transforms.functional_tensor which was removed
try:
    import torchvision
    import torchvision.transforms.functional as F
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

# =============================================================================
# Configuration
# =============================================================================

APP_TITLE = "Universal Media Studio"
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
CODEFORMER_DIR = MODELS_DIR / "CodeFormer"
REALESRGAN_BIN = BASE_DIR / "bin" / "realesrgan-ncnn-vulkan"
BIN_DIR = BASE_DIR / "bin"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables BEFORE importing heavy libraries
os.environ["PYTHONPATH"] = f"{BASE_DIR}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
os.environ["NLTK_DATA"] = str(MODELS_DIR / "nltk")
os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
os.environ["FACEXLIB_HOME"] = str(MODELS_DIR / "facexlib")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Now import heavy libraries
import whisperx
import nltk

# Check NVENC
HAS_NVENC = False
try:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    HAS_NVENC = "h264_nvenc" in r.stdout
except:
    pass


def cleanup_old_outputs(max_age_hours=24):
    """Remove output files older than max_age_hours."""
    if not OUTPUT_DIR.exists():
        return
    now = datetime.now().timestamp()
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            if (now - f.stat().st_mtime) > (max_age_hours * 3600):
                try:
                    f.unlink()
                except:
                    pass

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
    """Clear GPU cache and collect garbage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.synchronize()
        except:
            pass


# =============================================================================
# Media Presets - Returns (batch_size, upscale_model, face_weight)
# =============================================================================

PRESETS = {
    "Anime": (16, "realesrgan-x4plus-anime", 0.3),
    "Live Action": (16, "realesrgan-x4plus", 0.7),
    "Podcast": (24, "realesrgan-x4plus", 0.5),
}


# =============================================================================
# Backend Functions
# =============================================================================

def generate_subtitles(
    audio_file: str,
    model_size: str,
    batch_size: int,
    language: str,
    hf_token: str,
    enable_diarization: bool,
    progress: gr.Progress = gr.Progress()
) -> Tuple[str, Optional[str], Optional[str]]:
    """Generate subtitles using WhisperX with full feature utilization."""
    flush_vram()
    
    if not audio_file:
        raise gr.Error("Please upload an audio or video file.")
    
    if not Path(audio_file).exists():
        raise gr.Error(f"File not found: {audio_file}")
    
    try:
        progress(0.1, desc="Loading WhisperX...")
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        
        progress(0.2, desc="Loading audio...")
        audio = whisperx.load_audio(audio_file)
        
        progress(0.4, desc="Transcribing...")
        model = whisperx.load_model(
            model_size, DEVICE,
            compute_type=compute_type,
            download_root=str(WHISPER_DIR)
        )
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
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, 
            device=DEVICE,
            model_dir=str(MODELS_DIR / "alignment")
        )
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
        ass_header = """[Script Info]
Title: Universal Media Studio
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
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


def get_free_space_gb():
    """Check remaining disk space in GB."""
    stat = os.statvfs(BASE_DIR)
    return (stat.f_bavail * stat.f_frsize) / (1024**3)

def restore_video(
    video_file: str,
    target_res: str,
    upscale_model: str,
    tile_size: int,
    face_weight: float,
    enable_face: bool,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], Optional[str]]:
    """Upscale video using Real-ESRGAN binary and optionally CodeFormer."""
    flush_vram()
    
    if not video_file:
        return None, None

    # Disk space safety check
    free_gb = get_free_space_gb()
    if free_gb < 10:
        raise gr.Error(f"Insufficient disk space ({free_gb:.1f}GB remaining). At least 10GB required for frame extraction.")
    
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = Path(video_file).stem
        
        frames_dir = TEMP_DIR / f"frames_{ts}"
        upscaled_dir = TEMP_DIR / f"upscaled_{ts}"
        restored_dir = TEMP_DIR / f"restored_{ts}"
        
        for d in [frames_dir, upscaled_dir, restored_dir]:
            d.mkdir(exist_ok=True)
        
        progress(0.1, desc="Extracting frames...")
        
        # Get FPS
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,width,height", "-of", "json", video_file],
            capture_output=True, text=True, timeout=30
        )
        if probe.returncode != 0:
            raise gr.Error(f"Failed to analyze video: {probe.stderr}")
        info = json.loads(probe.stdout)
        stream_info = info["streams"][0]
        fps_parts = stream_info["r_frame_rate"].split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])
        input_height = int(stream_info.get("height", 480))
        
        # Extract frames (use %08d for long videos)
        extract_result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_file, "-qscale:v", "1", str(frames_dir / "frame_%08d.png")],
            capture_output=True, text=True, timeout=1200
        )
        if extract_result.returncode != 0:
            raise gr.Error(f"Frame extraction failed: {extract_result.stderr}")
        
        progress(0.3, desc="Upscaling with Real-ESRGAN...")
        
        # Calculate scale based on input resolution
        target_height = 2160 if target_res == "4K" else 1080
        scale = max(2, min(4, (target_height // input_height) or 2))
        
        if REALESRGAN_BIN.exists():
            esrgan_result = subprocess.run([
                str(REALESRGAN_BIN),
                "-i", str(frames_dir),
                "-o", str(upscaled_dir),
                "-s", str(scale),
                "-t", str(tile_size),
                "-n", upscale_model,
                "-f", "png"
            ], capture_output=True, text=True, cwd=str(BIN_DIR), timeout=3600)
            if esrgan_result.returncode != 0:
                raise gr.Error(f"Real-ESRGAN failed: {esrgan_result.stderr}")
        else:
            raise gr.Error(f"Real-ESRGAN binary not found at {REALESRGAN_BIN}")
        
        final_dir = upscaled_dir
        
        # CodeFormer face restoration
        if enable_face:
            cf_script = CODEFORMER_DIR / "inference_codeformer.py"
            if not cf_script.exists():
                raise gr.Error(
                    "Face restoration is enabled, but CodeFormer is not available in this environment. "
                    "If you're running in Docker/RunPod, restart the pod so start.py can fetch CodeFormer."
                )

            progress(0.6, desc="Restoring faces with CodeFormer...")
            try:
                subprocess.run([
                    sys.executable, str(cf_script),
                    "-w", str(face_weight),
                    "--input_path", str(upscaled_dir),
                    "--output_path", str(restored_dir),
                    "--bg_upsampler", "None"
                ], capture_output=True, text=True, check=True, cwd=str(CODEFORMER_DIR), timeout=3600)

                # CodeFormer creates a 'final_results' subfolder inside the output path
                # BUT if we specify --output_path, it might put files directly or in a subfolder
                # depending on the input type. For directory input, it usually does:
                # {output_path}/final_results/*.png
                cf_out = restored_dir / "final_results"
                if cf_out.exists() and any(cf_out.iterdir()):
                    final_dir = cf_out
                elif restored_dir.exists() and any(restored_dir.iterdir()):
                    # Fallback if it puts them directly in restored_dir
                    final_dir = restored_dir
                else:
                    raise gr.Error("CodeFormer completed but produced no output frames.")
            except subprocess.CalledProcessError as e:
                raise gr.Error(
                    "CodeFormer failed. "
                    f"stdout: {e.stdout}\n"
                    f"stderr: {e.stderr}"
                )
        
        progress(0.8, desc="Encoding video...")
        
        res = "3840:2160" if target_res == "4K" else "1920:1080"
        output_path = OUTPUT_DIR / f"{name}_restored_{ts}.mp4"
        
        # Verify upscaled frames exist
        upscaled_frames = sorted(upscaled_dir.glob("*.png"))
        if not upscaled_frames:
            raise gr.Error("Upscaling produced no output frames.")
        
        # Professional scaling: keep aspect ratio, pad with black, and ensure YUV420P
        vf = f"scale={res}:force_original_aspect_ratio=decrease,pad={res}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(final_dir / "frame_%08d.png"),
            "-i", video_file, "-map", "0:v:0", "-map", "1:a:0?",
            "-vf", vf
        ]
        
        if HAS_NVENC:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
        
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path)])
        
        encode_result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if encode_result.returncode != 0:
            raise gr.Error(f"Video encoding failed: {encode_result.stderr}")
        
        # Cleanup
        for d in [frames_dir, upscaled_dir, restored_dir]:
            shutil.rmtree(d, ignore_errors=True)
        
        progress(1.0, desc="Complete!")
        return str(output_path), str(output_path)
        
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Restoration failed: {str(e)}")
    finally:
        # Always clean up temp files
        for d in [frames_dir, upscaled_dir, restored_dir]:
            shutil.rmtree(d, ignore_errors=True)
        flush_vram()


def enhance_audio(audio_file: str, attenuation: float, progress: gr.Progress = gr.Progress()) -> Optional[str]:
    """Enhance audio using DeepFilterNet with configurable attenuation."""
    flush_vram()
    if not audio_file:
        raise gr.Error("Please upload an audio file.")
    
    if not Path(audio_file).exists():
        raise gr.Error(f"Audio file not found: {audio_file}")
    
    out_dir = None
    try:
        progress(0.2, desc="Running DeepFilterNet...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / f"enhanced_{ts}"
        out_dir.mkdir(exist_ok=True)
        
        # Check if df-enhance is available
        if shutil.which("df-enhance") is None:
            raise gr.Error("DeepFilterNet (df-enhance) is not installed or not in PATH.")

        # DeepFilterNet CLI with post-filter for stronger attenuation
        cmd = ["df-enhance", "--output-dir", str(out_dir)]
        if DEVICE == "cuda":
            cmd.extend(["--device", "0"]) # Explicitly target first GPU
        
        if attenuation > 0.7:
            cmd.append("--pf")  # Post-filter for aggressive noise reduction
        cmd.append(audio_file)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise gr.Error(f"DeepFilterNet failed: {result.stderr}")
        
        files = list(out_dir.glob("*.wav"))
        if not files:
            raise gr.Error("DeepFilterNet produced no output.")
        
        progress(1.0, desc="Complete!")
        return str(files[0])
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Audio enhancement failed: {str(e)}")
    finally:
        flush_vram()


def separate_stems(audio_file: str, model: str, progress: gr.Progress = gr.Progress()) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Separate audio stems using Demucs with model selection."""
    flush_vram()
    if not audio_file:
        raise gr.Error("Please upload an audio file.")
    
    if not Path(audio_file).exists():
        raise gr.Error(f"Audio file not found: {audio_file}")
    
    try:
        progress(0.1, desc=f"Running Demucs ({model})...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / f"stems_{ts}"
        
        # Check if Demucs is available
        import importlib.util
        if importlib.util.find_spec("demucs") is None:
            raise gr.Error("Demucs is not installed in the current environment.")

        cmd = [
            sys.executable, "-m", "demucs",
            "--out", str(out_dir), "-n", model, audio_file
        ]
        if DEVICE == "cuda":
            cmd.extend(["-d", "cuda"])
            
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200)
        
        name = Path(audio_file).stem
        stems_dir = out_dir / model / name
        
        if not stems_dir.exists():
            raise gr.Error(f"Demucs output not found at {stems_dir}")
        
        progress(1.0, desc="Complete!")
        
        def get_stem(s: str) -> Optional[str]:
            p = stems_dir / f"{s}.wav"
            return str(p) if p.exists() else None
        
        return get_stem("vocals"), get_stem("drums"), get_stem("bass"), get_stem("other")
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Stem separation failed: {str(e)}")
    finally:
        flush_vram()


def burn_subtitles(video: str, subs: Any, font_size: int, progress: gr.Progress = gr.Progress()) -> str:
    """Burn subtitles into video."""
    flush_vram()
    if not video:
        raise gr.Error("Please upload a video file.")
    if not subs:
        raise gr.Error("Please upload a subtitle file.")
    
    # CRITICAL: gr.File returns object with .name attribute for path
    sub_path = subs.name if hasattr(subs, 'name') else str(subs)
    
    if not Path(sub_path).exists():
        raise gr.Error(f"Subtitle file not found: {sub_path}")
    
    try:
        progress(0.2, desc="Burning subtitles...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"{Path(video).stem}_subtitled_{ts}.mp4"
        
        # Hardened escaping for FFmpeg filter paths (internal FFmpeg escaping)
        # For the subtitles/ass filters, the filename is a string within a filtergraph.
        # 1. Backslashes must be doubled
        # 2. Single quotes must be escaped as '\''
        # 3. Colons must be escaped if they are interpreted as separators
        escaped_path = str(sub_path).replace("\\", "/").replace("'", "'\\''").replace(":", "\\:")
        
        # Use 'ass' filter for both .ass and .ssa files
        if sub_path.lower().endswith((".ass", ".ssa")):
            vf = f"ass='{escaped_path}'"
        else:
            # force_style uses its own comma-separated key=value format
            vf = f"subtitles='{escaped_path}':force_style='FontSize={font_size}'"
        
        cmd = ["ffmpeg", "-y", "-i", video, "-vf", vf]
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"] if HAS_NVENC else ["-c:v", "libx264", "-crf", "18"])
        cmd.extend(["-c:a", "copy", "-movflags", "+faststart", str(out)])
        
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200)
        
        progress(1.0, desc="Complete!")
        return str(out)
    except Exception as e:
        raise gr.Error(f"FFmpeg failed: {str(e)}")
    finally:
        flush_vram()


def convert_video(video: str, fmt: str, vcodec: str, acodec: str, crf: int, progress: gr.Progress = gr.Progress()) -> Optional[str]:
    """Convert video format."""
    flush_vram()
    if not video:
        raise gr.Error("Please upload a video file.")
    
    if not Path(video).exists():
        raise gr.Error(f"Video file not found: {video}")
    
    try:
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


def extract_audio(video: str, fmt: str, progress: gr.Progress = gr.Progress()) -> Optional[str]:
    """Extract audio from video."""
    flush_vram()
    if not video:
        raise gr.Error("Please upload a video file.")
    
    if not Path(video).exists():
        raise gr.Error(f"Video file not found: {video}")
    
    try:
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


def on_preset_change(preset: str) -> Tuple[int, str, float]:
    """Update UI components based on selected media preset."""
    batch, model, face_w = PRESETS.get(preset, PRESETS["Live Action"])
    return batch, model, face_w


def process_audio(
    audio: str,
    denoise: bool,
    strength: float,
    separate: bool,
    model: str,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Combined helper for audio enhancement and stem separation."""
    if not audio:
        raise gr.Error("Please upload an audio file.")
    enhanced = None
    v, d, b, o = None, None, None, None
    try:
        if denoise:
            enhanced = enhance_audio(audio, strength, progress)
        if separate:
            v, d, b, o = separate_stems(audio, model, progress)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Audio processing failed: {str(e)}")
    return enhanced, v, d, b, o


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
                
                # Connect UI actions
                sub_btn.click(
                    fn=generate_subtitles,
                    inputs=[sub_input, sub_model, sub_batch, sub_lang, sub_token, sub_diarize],
                    outputs=[sub_transcript, sub_srt, sub_ass]
                )
                
                vid_btn.click(
                    fn=restore_video,
                    inputs=[vid_input, vid_res, vid_model, vid_tile, vid_face_w, vid_face_enable],
                    outputs=[vid_preview, vid_download]
                )
                
                # Preset connection
                preset_dropdown.change(
                    fn=on_preset_change,
                    inputs=[preset_dropdown],
                    outputs=[sub_batch, vid_model, vid_face_w]
                )
                
                # Toolbox tools
                burn_btn.click(burn_subtitles, [burn_vid, burn_sub, burn_font], burn_out)
                conv_btn.click(convert_video, [conv_in, conv_fmt, conv_vc, conv_ac, conv_crf], conv_out)
                ext_btn.click(extract_audio, [ext_in, ext_fmt], ext_out)
                at_btn.click(
                    process_audio,
                    [at_in, at_denoise, at_strength, at_separate, at_model],
                    [at_enhanced, at_vocals, at_drums, at_bass, at_other]
                )
        
        # Footer
        gr.Markdown(
            "---\n"
            "<center>"
            "<small>🎬 <b>Universal Media Studio</b> | "
            "WhisperX • Real-ESRGAN • CodeFormer • DeepFilterNet • Demucs</small>\n"
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
    app.queue(max_size=10, default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
