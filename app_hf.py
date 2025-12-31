#!/usr/bin/env python3
"""
Universal Media Studio - HuggingFace Spaces Version
Optimized for HF Spaces with ZeroGPU support.
"""

import gc
import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
import gradio as gr

# Check if running on HuggingFace Spaces
IS_HF_SPACES = os.environ.get("SPACE_ID") is not None

# Import spaces module for ZeroGPU (only available on HF)
if IS_HF_SPACES:
    import spaces

# =============================================================================
# Configuration
# =============================================================================

APP_TITLE = "Universal Media Studio"

# Use HF cache directories
if IS_HF_SPACES:
    CACHE_DIR = Path("/tmp/ums_cache")
    MODELS_DIR = Path(os.environ.get("HF_HOME", "/tmp/hf_cache"))
else:
    CACHE_DIR = Path(__file__).parent / "cache"
    MODELS_DIR = Path(__file__).parent / "models"

WHISPER_DIR = MODELS_DIR / "whisper"
OUTPUT_DIR = CACHE_DIR / "outputs"
TEMP_DIR = CACHE_DIR / "temp"

# Create directories
for d in [CACHE_DIR, MODELS_DIR, WHISPER_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Check NVENC (unlikely on HF Spaces, but check anyway)
HAS_NVENC = False
try:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5)
    HAS_NVENC = "h264_nvenc" in r.stdout
except:
    pass


def flush_vram():
    """Flush GPU memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Media Presets
# =============================================================================

PRESETS = {
    "Anime": (16, "realesrgan-x4plus-anime", 0.3),
    "Live Action": (16, "realesrgan-x4plus", 0.7),
    "Podcast": (24, "realesrgan-x4plus", 0.5),
}


def apply_preset(preset: str):
    """Return preset values for UI components."""
    batch, model, face_w = PRESETS.get(preset, PRESETS["Live Action"])
    return batch, model, face_w


# =============================================================================
# GPU-Decorated Functions (ZeroGPU compatible)
# =============================================================================

# Decorator that works on HF Spaces with ZeroGPU, no-op elsewhere
def gpu_decorator(duration=120):
    """Apply @spaces.GPU decorator only on HuggingFace Spaces."""
    if IS_HF_SPACES:
        return spaces.GPU(duration=duration)
    return lambda fn: fn


@gpu_decorator(duration=300)
def generate_subtitles(
    audio_file: str,
    model_size: str,
    batch_size: int,
    language: str,
    hf_token: str,
    enable_diarization: bool,
    progress: gr.Progress = gr.Progress()
) -> Tuple[str, Optional[str], Optional[str]]:
    """Generate subtitles using WhisperX with ZeroGPU support."""
    flush_vram()
    
    if not audio_file:
        raise gr.Error("Please upload an audio or video file.")
    
    if not Path(audio_file).exists():
        raise gr.Error(f"File not found: {audio_file}")
    
    try:
        import whisperx
        
        progress(0.1, desc="Loading WhisperX...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = whisperx.load_model(
            model_size, device,
            compute_type=compute_type,
            download_root=str(WHISPER_DIR),
            language=language if language != "auto" else None
        )
        
        progress(0.2, desc="Loading audio...")
        audio = whisperx.load_audio(audio_file)
        
        progress(0.3, desc="Transcribing...")
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            language=language if language != "auto" else None
        )
        detected_lang = result["language"]
        
        del model
        flush_vram()
        
        progress(0.5, desc="Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
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
            progress(0.7, desc="Diarizing speakers...")
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
                flush_vram()
            except Exception as e:
                raise gr.Error(f"Diarization failed: {e}\nCheck your HuggingFace token and model agreements.")
        
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
        flush_vram()
        raise gr.Error(f"Transcription failed: {str(e)}")


@gpu_decorator(duration=120)
def enhance_audio(audio_file: str, attenuation: float, progress: gr.Progress = gr.Progress()) -> Optional[str]:
    """Enhance audio using DeepFilterNet."""
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
        
        cmd = ["deepFilter", "--output-dir", str(out_dir)]
        if attenuation > 0.7:
            cmd.append("--pf")
        cmd.append(audio_file)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
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


@gpu_decorator(duration=300)
def separate_stems(audio_file: str, model: str, progress: gr.Progress = gr.Progress()):
    """Separate audio stems using Demucs."""
    flush_vram()
    if not audio_file:
        raise gr.Error("Please upload an audio file.")
    
    if not Path(audio_file).exists():
        raise gr.Error(f"Audio file not found: {audio_file}")
    
    try:
        progress(0.1, desc=f"Running Demucs ({model})...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / f"stems_{ts}"
        
        result = subprocess.run([
            sys.executable, "-m", "demucs",
            "--out", str(out_dir), "-n", model, audio_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise gr.Error(f"Demucs failed: {result.stderr}")
        
        name = Path(audio_file).stem
        stems_dir = out_dir / model / name
        
        if not stems_dir.exists():
            raise gr.Error(f"Demucs output not found at {stems_dir}")
        
        progress(1.0, desc="Complete!")
        
        def get_stem(s):
            p = stems_dir / f"{s}.wav"
            return str(p) if p.exists() else None
        
        return get_stem("vocals"), get_stem("drums"), get_stem("bass"), get_stem("other")
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Stem separation failed: {str(e)}")
    finally:
        flush_vram()


def burn_subtitles(video: str, subs, font_size: int, progress: gr.Progress = gr.Progress()):
    """Burn subtitles into video (CPU operation)."""
    if not video:
        raise gr.Error("Please upload a video file.")
    if not subs:
        raise gr.Error("Please upload a subtitle file.")
    
    sub_path = subs.name if hasattr(subs, 'name') else str(subs)
    
    if not Path(sub_path).exists():
        raise gr.Error(f"Subtitle file not found: {sub_path}")
    
    progress(0.2, desc="Burning subtitles...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"{Path(video).stem}_subtitled_{ts}.mp4"
    
    escaped_path = str(sub_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"ass='{escaped_path}'" if sub_path.endswith(".ass") else f"subtitles='{escaped_path}':force_style='FontSize={font_size}'"
    
    # Use software encoding on HF Spaces
    cmd = ["ffmpeg", "-y", "-i", video, "-vf", vf, "-c:v", "libx264", "-crf", "18", "-c:a", "copy", "-movflags", "+faststart", str(out)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"FFmpeg failed: {e.stderr if e.stderr else str(e)}")
    
    progress(1.0, desc="Complete!")
    return str(out)


def convert_video(video: str, fmt: str, vcodec: str, acodec: str, crf: int, progress: gr.Progress = gr.Progress()):
    """Convert video format (CPU operation)."""
    if not video:
        raise gr.Error("Please upload a video file.")
    
    if not Path(video).exists():
        raise gr.Error(f"Video file not found: {video}")
    
    try:
        progress(0.2, desc="Converting...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"{Path(video).stem}_converted_{ts}.{fmt}"
        
        cmd = ["ffmpeg", "-y", "-i", video]
        
        # Map codecs (no NVENC on HF Spaces typically)
        vc_map = {
            "H.264": ["-c:v", "libx264", "-crf", str(crf)],
            "H.265": ["-c:v", "libx265", "-crf", str(crf)],
            "VP9": ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0"],
            "Copy": ["-c:v", "copy"]
        }
        cmd.extend(vc_map.get(vcodec, ["-c:v", "libx264", "-crf", str(crf)]))
        
        ac_map = {"AAC": ["-c:a", "aac", "-b:a", "192k"], "Opus": ["-c:a", "libopus", "-b:a", "128k"], "Copy": ["-c:a", "copy"]}
        cmd.extend(ac_map.get(acodec, ["-c:a", "aac"]))
        cmd.extend(["-movflags", "+faststart", str(out)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise gr.Error(f"FFmpeg conversion failed: {result.stderr}")
        
        progress(1.0, desc="Complete!")
        return str(out)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Video conversion failed: {str(e)}")


def extract_audio(video: str, fmt: str, progress: gr.Progress = gr.Progress()):
    """Extract audio from video (CPU operation)."""
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
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise gr.Error(f"Audio extraction failed: {result.stderr}")
        
        progress(1.0, desc="Complete!")
        return str(out)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Audio extraction failed: {str(e)}")


# =============================================================================
# Gradio UI
# =============================================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
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
    # Detect environment
    if IS_HF_SPACES:
        env_badge = "<span class='status-badge status-success'>🤗 HuggingFace Spaces</span>"
        gpu_info = "<span class='status-badge status-success'>ZeroGPU</span>" if torch.cuda.is_available() else "<span class='status-badge status-warning'>CPU</span>"
    else:
        gpu_info = f"<span class='status-badge status-success'>✓ {DEVICE.upper()}</span>" if DEVICE == "cuda" else "<span class='status-badge status-warning'>CPU Mode</span>"
        env_badge = ""
    
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono")
        ),
        css=CUSTOM_CSS
    ) as app:
        
        gr.Markdown(f"""
        # 🎬 {APP_TITLE}
        **AI-Powered Media Processing**
        
        {env_badge} {gpu_info}
        """)
        
        # Preset Bar
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()), value="Live Action",
                label="🎯 Media Preset",
                info="Auto-configures optimal settings",
                scale=2,
                interactive=True
            )
            gr.Markdown(
                "<div style='padding: 10px; background: #f1f5f9; border-radius: 8px; font-size: 0.9em;'>"
                "💡 <b>Tip:</b> Select a preset to configure AI models for your content type."
                "</div>",
                scale=4
            )
        
        # =================================================================
        # Tabs
        # =================================================================
        with gr.Tabs():
            # Tab 1: AI Subtitles
            with gr.TabItem("🎤 AI Subtitles", id="subtitles"):
                gr.Markdown(
                    "### Generate accurate, time-aligned subtitles\n"
                    "*Powered by WhisperX with word-level alignment*"
                )
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        sub_input = gr.Audio(
                            label="📁 Upload Audio/Video",
                            type="filepath",
                            sources=["upload", "microphone"],
                            show_download_button=False
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
                                info="Required for diarization"
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
                
                sub_btn.click(
                    fn=generate_subtitles,
                    inputs=[sub_input, sub_model, sub_batch, sub_lang, sub_token, sub_diarize],
                    outputs=[sub_transcript, sub_srt, sub_ass]
                )
            
            # Tab 2: Audio Tools
            with gr.TabItem("🎧 Audio Tools", id="audio"):
                gr.Markdown(
                    "### Enhance and separate audio\n"
                    "*DeepFilterNet for noise reduction, Demucs for stem separation*"
                )
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        at_in = gr.Audio(
                            label="📁 Input Audio",
                            type="filepath",
                            sources=["upload"]
                        )
                        gr.Markdown("##### 🔊 Noise Reduction")
                        at_denoise = gr.Checkbox(
                            label="Enable DeepFilterNet",
                            value=True,
                            info="AI-powered noise reduction"
                        )
                        at_strength = gr.Slider(
                            0.3, 1.0, value=0.7, step=0.1,
                            label="Denoise Strength",
                            info="Higher = more aggressive"
                        )
                        gr.Markdown("##### 🎸 Stem Separation")
                        at_separate = gr.Checkbox(
                            label="Enable Demucs",
                            value=False,
                            info="Separate vocals, drums, bass, other"
                        )
                        at_model = gr.Dropdown(
                            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"],
                            value="htdemucs",
                            label="Demucs Model",
                            info="htdemucs_ft is fine-tuned for music"
                        )
                        at_btn = gr.Button("🎯 Process Audio", variant="primary", elem_classes=["primary-btn"])
                    
                    with gr.Column(scale=1, elem_classes=["output-panel"]):
                        gr.Markdown("#### 📤 Output")
                        at_enhanced = gr.Audio(label="✨ Enhanced", type="filepath")
                        gr.Markdown("##### Separated Stems")
                        with gr.Row():
                            at_vocals = gr.Audio(label="🎤 Vocals", type="filepath")
                            at_other = gr.Audio(label="🎸 Instrumental", type="filepath")
                        with gr.Row():
                            at_drums = gr.Audio(label="🥁 Drums", type="filepath")
                            at_bass = gr.Audio(label="🎸 Bass", type="filepath")
                
                def process_audio(audio, denoise, strength, separate, model, progress=gr.Progress()):
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
                
                at_btn.click(
                    process_audio,
                    [at_in, at_denoise, at_strength, at_separate, at_model],
                    [at_enhanced, at_vocals, at_drums, at_bass, at_other]
                )
            
            # Tab 3: FFmpeg Toolbox
            with gr.TabItem("🛠️ Toolbox", id="toolbox"):
                gr.Markdown(
                    "### FFmpeg-powered utilities\n"
                    "*Convert, extract, and process media*"
                )
                
                with gr.Tabs():
                    with gr.TabItem("📝 Burn Subtitles"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                burn_vid = gr.Video(label="📁 Video", sources=["upload"])
                                burn_sub = gr.File(
                                    label="📄 Subtitle File",
                                    file_types=[".srt", ".ass", ".ssa", ".vtt"],
                                    file_count="single"
                                )
                                burn_font = gr.Slider(16, 48, value=24, step=2, label="Font Size")
                                burn_btn = gr.Button("🔥 Burn Subtitles", variant="primary")
                            with gr.Column(scale=1, elem_classes=["output-panel"]):
                                gr.Markdown("#### 📤 Output")
                                burn_out = gr.Video(label="📥 Result")
                        
                        burn_btn.click(burn_subtitles, [burn_vid, burn_sub, burn_font], burn_out)
                    
                    with gr.TabItem("🔄 Convert"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                conv_in = gr.Video(label="📁 Input Video", sources=["upload"])
                                conv_fmt = gr.Dropdown(["mp4", "mkv", "webm", "mov"], value="mp4", label="Format")
                                conv_vc = gr.Dropdown(
                                    ["H.264", "H.265", "VP9", "Copy"],
                                    value="H.264",
                                    label="Video Codec"
                                )
                                conv_ac = gr.Dropdown(["AAC", "Opus", "Copy"], value="AAC", label="Audio Codec")
                                conv_crf = gr.Slider(15, 35, value=20, step=1, label="Quality (CRF)")
                                conv_btn = gr.Button("⚡ Convert", variant="primary")
                            with gr.Column(scale=1, elem_classes=["output-panel"]):
                                gr.Markdown("#### 📤 Output")
                                conv_out = gr.File(label="📥 Converted File", file_count="single")
                        
                        conv_btn.click(convert_video, [conv_in, conv_fmt, conv_vc, conv_ac, conv_crf], conv_out)
                    
                    with gr.TabItem("🎵 Extract Audio"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                ext_in = gr.Video(label="📁 Input Video", sources=["upload"])
                                ext_fmt = gr.Dropdown(["mp3", "wav", "flac", "aac", "opus"], value="mp3", label="Format")
                                ext_btn = gr.Button("🎶 Extract Audio", variant="primary")
                            with gr.Column(scale=1, elem_classes=["output-panel"]):
                                gr.Markdown("#### 📤 Output")
                                ext_out = gr.Audio(label="🔊 Extracted Audio", type="filepath")
                        
                        ext_btn.click(extract_audio, [ext_in, ext_fmt], ext_out)
        
        # Preset change handler
        def on_preset_change(preset):
            batch, model, face_w = PRESETS.get(preset, PRESETS["Live Action"])
            return batch
        
        preset_dropdown.change(fn=on_preset_change, inputs=[preset_dropdown], outputs=[sub_batch])
        
        # Footer
        gr.Markdown(
            "---\n"
            "<center>"
            "<small>🎬 <b>Universal Media Studio</b> | "
            "WhisperX • DeepFilterNet • Demucs • FFmpeg</small>\n"
            "<small style='color: #64748b;'>Powered by HuggingFace Spaces</small>"
            "</center>"
        )
    
    return app


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    app = create_ui()
    app.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
