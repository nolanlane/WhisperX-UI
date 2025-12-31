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
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
import gradio as gr

# =============================================================================
# Configuration
# =============================================================================

APP_TITLE = "Universal Media Studio"
BASE_DIR = Path("/app") if Path("/app").exists() else Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
CODEFORMER_DIR = BASE_DIR / "CodeFormer"
REALESRGAN_BIN = BASE_DIR / "bin" / "realesrgan-ncnn-vulkan"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Check NVENC
HAS_NVENC = False
try:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
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
# Media Presets - Returns (batch_size, upscale_model, face_weight)
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
        return "❌ Please upload a file.", None, None
    
    try:
        import whisperx
        
        progress(0.1, desc="Loading WhisperX...")
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        
        model = whisperx.load_model(
            model_size, DEVICE,
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
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
        
        del model_a
        flush_vram()
        
        # Speaker diarization
        if enable_diarization and hf_token:
            progress(0.7, desc="Diarizing speakers...")
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
                flush_vram()
            except Exception as e:
                pass  # Continue without diarization
        
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
        
    except Exception as e:
        flush_vram()
        return f"❌ Error: {e}", None, None


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
             "-show_entries", "stream=r_frame_rate", "-of", "json", video_file],
            capture_output=True, text=True
        )
        info = json.loads(probe.stdout)
        fps_parts = info["streams"][0]["r_frame_rate"].split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        # Extract frames (use %08d for long videos)
        extract_result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_file, "-qscale:v", "1", str(frames_dir / "frame_%08d.png")],
            capture_output=True, text=True
        )
        if extract_result.returncode != 0:
            raise gr.Error(f"Frame extraction failed: {extract_result.stderr}")
        
        progress(0.3, desc="Upscaling with Real-ESRGAN...")
        
        # Calculate scale based on input resolution
        target_height = 2160 if target_res == "4K" else 1080
        scale = max(2, min(4, (target_height // int(info["streams"][0].get("height", 480))) or 2))
        
        if REALESRGAN_BIN.exists():
            esrgan_result = subprocess.run([
                str(REALESRGAN_BIN),
                "-i", str(frames_dir),
                "-o", str(upscaled_dir),
                "-s", str(scale),
                "-t", str(tile_size),
                "-n", upscale_model,
                "-f", "png"
            ], capture_output=True, text=True)
            if esrgan_result.returncode != 0:
                raise gr.Error(f"Real-ESRGAN failed: {esrgan_result.stderr}")
        else:
            raise gr.Error(f"Real-ESRGAN binary not found at {REALESRGAN_BIN}")
        
        final_dir = upscaled_dir
        
        # CodeFormer face restoration
        if enable_face and CODEFORMER_DIR.exists():
            progress(0.6, desc="Restoring faces with CodeFormer...")
            cf_script = CODEFORMER_DIR / "inference_codeformer.py"
            if cf_script.exists():
                try:
                    subprocess.run([
                        sys.executable, str(cf_script),
                        "-w", str(face_weight),
                        "--input_path", str(upscaled_dir),
                        "--output_path", str(restored_dir),
                        "--bg_upsampler", "None"
                    ], capture_output=True, check=True, cwd=str(CODEFORMER_DIR))
                    
                    cf_out = restored_dir / "final_results"
                    if cf_out.exists() and any(cf_out.iterdir()):
                        final_dir = cf_out
                except:
                    pass
        
        progress(0.8, desc="Encoding video...")
        
        res = "3840:2160" if target_res == "4K" else "1920:1080"
        output_path = OUTPUT_DIR / f"{name}_restored_{ts}.mp4"
        
        # Verify upscaled frames exist
        upscaled_frames = sorted(upscaled_dir.glob("*.png"))
        if not upscaled_frames:
            raise gr.Error("Upscaling produced no output frames.")
        
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(final_dir / "frame_%08d.png"),
            "-i", video_file, "-map", "0:v:0", "-map", "1:a:0?",
            "-vf", f"scale={res}:flags=lanczos"
        ]
        
        if HAS_NVENC:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
        
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path)])
        
        encode_result = subprocess.run(cmd, capture_output=True, text=True)
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
        return None
    
    try:
        progress(0.2, desc="Running DeepFilterNet...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / f"enhanced_{ts}"
        out_dir.mkdir(exist_ok=True)
        
        # DeepFilterNet CLI with post-filter for stronger attenuation
        cmd = ["deepFilter", "--output-dir", str(out_dir)]
        if attenuation > 0.7:
            cmd.append("--pf")  # Post-filter for aggressive noise reduction
        cmd.append(audio_file)
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        files = list(out_dir.glob("*.wav"))
        progress(1.0, desc="Complete!")
        return str(files[0]) if files else None
    except Exception as e:
        raise gr.Error(f"Enhancement failed: {e}")


def separate_stems(audio_file: str, model: str, progress: gr.Progress = gr.Progress()):
    """Separate audio stems using Demucs with model selection."""
    flush_vram()
    if not audio_file:
        return None, None, None, None
    
    try:
        progress(0.1, desc=f"Running Demucs ({model})...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / f"stems_{ts}"
        
        subprocess.run([
            sys.executable, "-m", "demucs",
            "--out", str(out_dir), "-n", model, audio_file
        ], capture_output=True, check=True)
        
        name = Path(audio_file).stem
        stems_dir = out_dir / model / name
        
        progress(1.0, desc="Complete!")
        
        def get_stem(s):
            p = stems_dir / f"{s}.wav"
            return str(p) if p.exists() else None
        
        return get_stem("vocals"), get_stem("drums"), get_stem("bass"), get_stem("other")
    except Exception as e:
        raise gr.Error(f"Separation failed: {e}")


def burn_subtitles(video: str, subs, font_size: int, progress: gr.Progress = gr.Progress()):
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
    
    progress(0.2, desc="Burning subtitles...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"{Path(video).stem}_subtitled_{ts}.mp4"
    
    # Escape path for ffmpeg filter
    escaped_path = str(sub_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"ass='{escaped_path}'" if sub_path.endswith(".ass") else f"subtitles='{escaped_path}':force_style='FontSize={font_size}'"
    
    cmd = ["ffmpeg", "-y", "-i", video, "-vf", vf]
    cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"] if HAS_NVENC else ["-c:v", "libx264", "-crf", "18"])
    cmd.extend(["-c:a", "copy", "-movflags", "+faststart", str(out)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"FFmpeg failed: {e.stderr if e.stderr else str(e)}")
    
    progress(1.0, desc="Complete!")
    return str(out)


def convert_video(video: str, fmt: str, vcodec: str, acodec: str, crf: int, progress: gr.Progress = gr.Progress()):
    """Convert video format."""
    flush_vram()
    if not video:
        return None
    
    progress(0.2, desc="Converting...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"{Path(video).stem}_converted_{ts}.{fmt}"
    
    cmd = ["ffmpeg", "-y", "-i", video]
    
    vc_map = {
        "H.264": ["-c:v", "libx264", "-crf", str(crf)],
        "H.265": ["-c:v", "libx265", "-crf", str(crf)],
        "H.264 (NVENC)": ["-c:v", "h264_nvenc", "-cq", str(crf)],
        "H.265 (NVENC)": ["-c:v", "hevc_nvenc", "-cq", str(crf)],
        "VP9": ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0"],
        "Copy": ["-c:v", "copy"]
    }
    cmd.extend(vc_map.get(vcodec, ["-c:v", "libx264", "-crf", str(crf)]))
    
    ac_map = {"AAC": ["-c:a", "aac", "-b:a", "192k"], "Opus": ["-c:a", "libopus", "-b:a", "128k"], "Copy": ["-c:a", "copy"]}
    cmd.extend(ac_map.get(acodec, ["-c:a", "aac"]))
    cmd.append(str(out))
    
    subprocess.run(cmd, capture_output=True, check=True)
    progress(1.0, desc="Complete!")
    return str(out)


def extract_audio(video: str, fmt: str, progress: gr.Progress = gr.Progress()):
    """Extract audio from video."""
    flush_vram()
    if not video:
        return None
    
    progress(0.2, desc="Extracting...")
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
    subprocess.run(cmd, capture_output=True, check=True)
    progress(1.0, desc="Complete!")
    return str(out)


# =============================================================================
# Gradio UI - Clean Dashboard Layout
# =============================================================================

def create_ui():
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", font=gr.themes.GoogleFont("Inter")),
        css=".gradio-container{max-width:1400px!important} footer{display:none!important}"
    ) as app:
        
        # Header
        gr.Markdown(f"# 🎬 {APP_TITLE}\n**AI-Powered Media Processing** | GPU: `{DEVICE.upper()}` | NVENC: `{'✓' if HAS_NVENC else '✗'}`")
        
        # =====================================================================
        # Global Preset Bar - Actually updates UI components
        # =====================================================================
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()), value="Live Action",
                label="🎯 Media Preset", info="Auto-configures settings", scale=2
            )
            gr.Markdown("", scale=6)
        
        # Hidden state components that get updated by preset
        batch_state = gr.State(value=16)
        model_state = gr.State(value="realesrgan-x4plus")
        face_state = gr.State(value=0.7)
        
        # =====================================================================
        # Tab 1: AI Subtitles
        # =====================================================================
        with gr.Tabs():
            with gr.TabItem("🎤 AI Subtitles"):
                gr.Markdown("### Generate accurate, time-aligned subtitles")
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        sub_input = gr.Audio(label="📁 Upload Audio/Video", type="filepath", sources=["upload", "microphone"])
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            sub_model = gr.Dropdown(
                                choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                                value="large-v3", label="Whisper Model"
                            )
                            sub_batch = gr.Slider(1, 32, value=16, step=1, label="Batch Size")
                            sub_lang = gr.Dropdown(
                                choices=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
                                value="auto", label="Language", info="Auto-detect or specify"
                            )
                            sub_diarize = gr.Checkbox(label="Speaker Diarization", value=False)
                            sub_token = gr.Textbox(label="HuggingFace Token", type="password", placeholder="hf_...")
                        
                        sub_btn = gr.Button("🚀 Generate Subtitles", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        sub_transcript = gr.Textbox(label="📝 Transcript", lines=12, interactive=True, show_copy_button=True)
                        with gr.Row():
                            sub_srt = gr.File(label="📄 SRT")
                            sub_ass = gr.File(label="📄 ASS")
                
                sub_btn.click(
                    fn=generate_subtitles,
                    inputs=[sub_input, sub_model, sub_batch, sub_lang, sub_token, sub_diarize],
                    outputs=[sub_transcript, sub_srt, sub_ass]
                )
            
            # =================================================================
            # Tab 2: Visual Restoration
            # =================================================================
            with gr.TabItem("🖼️ Visual Restoration"):
                gr.Markdown("### Upscale and enhance video with AI")
                
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        vid_input = gr.Video(label="📁 Upload Video", sources=["upload"])
                        vid_res = gr.Radio(["1080p", "4K"], value="1080p", label="🎯 Target Resolution")
                        
                        with gr.Accordion("🔧 Tweak Upscaler", open=False):
                            vid_model = gr.Dropdown(
                                choices=["realesrgan-x4plus", "realesrgan-x4plus-anime"],
                                value="realesrgan-x4plus", label="Upscale Model"
                            )
                            vid_tile = gr.Slider(32, 512, value=256, step=32, label="Tile Size", info="Lower = less VRAM")
                            vid_face_enable = gr.Checkbox(label="Face Restoration (CodeFormer)", value=True)
                            vid_face_w = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Face Fidelity")
                        
                        # Audio Enhancement (integrated here per model usage)
                        with gr.Accordion("🎧 Audio Enhancement", open=False):
                            aud_enhance = gr.Checkbox(label="Denoise Audio (DeepFilterNet)", value=False)
                            aud_strength = gr.Slider(0.3, 1.0, value=0.7, step=0.1, label="Denoise Strength")
                            aud_separate = gr.Checkbox(label="Separate Vocals (Demucs)", value=False)
                            aud_demucs_model = gr.Dropdown(
                                choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"],
                                value="htdemucs", label="Demucs Model"
                            )
                        
                        vid_btn = gr.Button("✨ Restore Video", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        vid_preview = gr.Video(label="🎬 Preview", interactive=False)
                        vid_download = gr.File(label="📥 Download")
                        
                        # Audio outputs (shown when audio processing enabled)
                        with gr.Accordion("🎵 Audio Outputs", open=False):
                            aud_enhanced = gr.Audio(label="Enhanced Audio", type="filepath")
                            aud_vocals = gr.Audio(label="Vocals", type="filepath")
                            aud_other = gr.Audio(label="Instrumental", type="filepath")
                
                # Connect preset to update components
                def on_preset_change(preset):
                    batch, model, face_w = PRESETS.get(preset, PRESETS["Live Action"])
                    return batch, model, face_w, model, face_w
                
                preset_dropdown.change(
                    fn=on_preset_change,
                    inputs=[preset_dropdown],
                    outputs=[sub_batch, vid_model, vid_face_w, model_state, face_state]
                )
                
                vid_btn.click(
                    fn=restore_video,
                    inputs=[vid_input, vid_res, vid_model, vid_tile, vid_face_w, vid_face_enable],
                    outputs=[vid_preview, vid_download]
                )
            
            # =================================================================
            # Tab 3: The Toolbox (FFmpeg)
            # =================================================================
            with gr.TabItem("🛠️ Toolbox"):
                gr.Markdown("### FFmpeg-powered utilities")
                
                with gr.Tabs():
                    # Burn Subtitles
                    with gr.TabItem("📝 Burn Subtitles"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                burn_vid = gr.Video(label="📁 Video", sources=["upload"])
                                burn_sub = gr.File(label="📄 Subtitle (SRT/ASS)", file_types=[".srt", ".ass", ".ssa", ".vtt"])
                                burn_font = gr.Slider(16, 48, value=24, step=2, label="Font Size")
                                burn_btn = gr.Button("🔥 Burn Subtitles", variant="primary")
                            with gr.Column(scale=1):
                                burn_out = gr.Video(label="📥 Output")
                        
                        burn_btn.click(burn_subtitles, [burn_vid, burn_sub, burn_font], burn_out)
                    
                    # Convert
                    with gr.TabItem("🔄 Convert"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                conv_in = gr.Video(label="📁 Input", sources=["upload"])
                                conv_fmt = gr.Dropdown(["mp4", "mkv", "webm", "mov"], value="mp4", label="Format")
                                conv_vc = gr.Dropdown(
                                    ["H.264", "H.265", "H.264 (NVENC)", "H.265 (NVENC)", "VP9", "Copy"],
                                    value="H.264 (NVENC)" if HAS_NVENC else "H.264", label="Video Codec"
                                )
                                conv_ac = gr.Dropdown(["AAC", "Opus", "Copy"], value="AAC", label="Audio Codec")
                                conv_crf = gr.Slider(15, 35, value=20, step=1, label="Quality (CRF)")
                                conv_btn = gr.Button("⚡ Convert", variant="primary")
                            with gr.Column(scale=1):
                                conv_out = gr.File(label="📥 Output")
                        
                        conv_btn.click(convert_video, [conv_in, conv_fmt, conv_vc, conv_ac, conv_crf], conv_out)
                    
                    # Extract Audio
                    with gr.TabItem("🎵 Extract Audio"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                ext_in = gr.Video(label="📁 Input", sources=["upload"])
                                ext_fmt = gr.Dropdown(["mp3", "wav", "flac", "aac", "opus"], value="mp3", label="Format")
                                ext_btn = gr.Button("🎶 Extract", variant="primary")
                            with gr.Column(scale=1):
                                ext_out = gr.Audio(label="🔊 Output", type="filepath")
                        
                        ext_btn.click(extract_audio, [ext_in, ext_fmt], ext_out)
                    
                    # Standalone Audio Tools
                    with gr.TabItem("🎧 Audio Tools"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                at_in = gr.Audio(label="📁 Input Audio", type="filepath")
                                at_denoise = gr.Checkbox(label="Denoise (DeepFilterNet)", value=True)
                                at_strength = gr.Slider(0.3, 1.0, value=0.7, step=0.1, label="Strength")
                                at_separate = gr.Checkbox(label="Separate Stems (Demucs)", value=False)
                                at_model = gr.Dropdown(["htdemucs", "htdemucs_ft", "mdx_extra"], value="htdemucs", label="Model")
                                at_btn = gr.Button("🎯 Process Audio", variant="primary")
                            with gr.Column(scale=1):
                                at_enhanced = gr.Audio(label="Enhanced", type="filepath")
                                at_vocals = gr.Audio(label="Vocals", type="filepath")
                                at_drums = gr.Audio(label="Drums", type="filepath")
                                at_bass = gr.Audio(label="Bass", type="filepath")
                                at_other = gr.Audio(label="Other", type="filepath")
                        
                        def process_audio(audio, denoise, strength, separate, model, progress=gr.Progress()):
                            enhanced = enhance_audio(audio, strength, progress) if denoise else None
                            v, d, b, o = separate_stems(audio, model, progress) if separate else (None, None, None, None)
                            return enhanced, v, d, b, o
                        
                        at_btn.click(process_audio, [at_in, at_denoise, at_strength, at_separate, at_model],
                                    [at_enhanced, at_vocals, at_drums, at_bass, at_other])
        
        gr.Markdown("---\n<center><small>Universal Media Studio | WhisperX • Real-ESRGAN • CodeFormer • Demucs • DeepFilterNet</small></center>")
    
    return app


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*50}\n🎬 {APP_TITLE}\nDevice: {DEVICE} | NVENC: {HAS_NVENC}\n{'='*50}\n")
    app = create_ui()
    app.queue(max_size=10, default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
