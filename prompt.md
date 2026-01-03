# Universal Media Studio - Production Specification v2.0

You are a Principal Full-Stack AI Engineer. Generate **complete, error-free, production-ready** code for a media processing application that runs on RunPod with NVIDIA GPUs.

---

## CRITICAL RULES (MUST FOLLOW)

1. **Never assume Gradio component values are strings** - Use `.name` for file paths from `gr.File`. Components like `gr.Audio(type="filepath")` and `gr.Video` return string paths directly.
2. **Every subprocess call MUST have error handling** with user-friendly `gr.Error()` messages.
3. **All external binaries must be validated at startup** before the UI loads.
4. **Clean up temp files in `finally` blocks**, not just on success.
5. **Test every file path exists before using it**.
6. **Delete models after use** with `del model` followed by `flush_vram()` to free GPU memory.

---

## 1. AI Stack (Exact Implementations)

| Tool | Implementation | CRITICAL Notes |
|------|----------------|----------------|
| **WhisperX** | Python library (v3.7.4) | Use `compute_type="float16"` on GPU, `"int8"` on CPU. Delete model after use. |
| **PyTorch** | v2.8.0 (cu128) | Optimized for January 2026 stack. |
| **Gradio** | v5.0.0+ | Modern dashboard layout. No legacy v4 patches. |
| **Demucs** | `subprocess` | Run via `python -m demucs`. Memory-isolated from main process. |
| **Real-ESRGAN** | `realesrgan-ncnn-vulkan` binary | Download Linux binary from GitHub releases v0.2.0. |
| **CodeFormer** | `subprocess` calling `inference_codeformer.py` | Clone repo, install via `python basicsr/setup.py develop`. |
| **DeepFilterNet** | CLI `df-enhance` | Installed via pip as `deepfilternet`. |
| **FFmpeg** | Direct subprocess | Use `h264_nvenc` or `av1_nvenc` when available. |

---

## 2. Project Structure (Exact Files Required)

```
WhisperX-UI/
├── app.py                  # Main Gradio application (v5.0.0+)
├── download_models.py      # Build-time model downloader with S3/GitHub fallbacks
├── start.py                # Startup validation script (runs before app.py)
├── sitecustomize.py        # Global patches (NumPy 2.0, BasicSR, Gradio Client)
├── requirements.txt        # Pinned dependencies (v3.7.4 stack)
├── Dockerfile              # Production container config (CUDA 12.8)
├── .dockerignore           # Exclude __pycache__, .git, *.pyc, temp/, outputs/
└── README.md               # Setup and usage instructions
```

---

## 3. Startup Validation Script (`start.py`)

**This script MUST run before `app.py` to validate all dependencies:**

```python
#!/usr/bin/env python3
"""Validate all dependencies before starting the app."""
import sys
import os
import subprocess
from pathlib import Path

# Ensure sitecustomize is loaded
BASE_DIR = Path(__file__).parent
os.environ["PYTHONPATH"] = f"{BASE_DIR}:{os.environ.get('PYTHONPATH', '')}"

REQUIRED_BINARIES = [
    ("ffmpeg", ["ffmpeg", "-version"]),
    ("ffprobe", ["ffprobe", "-version"]),
]

OPTIONAL_BINARIES = [
    ("df-enhance", ["df-enhance", "--help"]),
]

REQUIRED_PATHS = [
    Path("/app/bin/realesrgan-ncnn-vulkan"),
    Path("/app/models/CodeFormer/inference_codeformer.py"),
]

def check_binary(name, cmd):
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False

def main():
    print("=" * 50)
    print("Universal Media Studio - Startup Validation")
    print("=" * 50)
    
    errors = []
    warnings = []
    
    # Check required binaries
    for name, cmd in REQUIRED_BINARIES:
        if check_binary(name, cmd):
            print(f"✓ {name}")
        else:
            print(f"✗ {name} - REQUIRED")
            errors.append(name)
    
    # Check optional binaries
    for name, cmd in OPTIONAL_BINARIES:
        if check_binary(name, cmd):
            print(f"✓ {name}")
        else:
            print(f"⚠ {name} - optional, some features disabled")
            warnings.append(name)
    
    # Check required paths
    for path in REQUIRED_PATHS:
        if path.exists():
            print(f"✓ {path.name}")
        else:
            print(f"⚠ {path} not found")
            warnings.append(str(path))
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ CUDA: {gpu_name} ({vram:.1f} GB)")
        else:
            print("⚠ CUDA not available - CPU mode (slower)")
            warnings.append("CUDA")
    except Exception as e:
        print(f"⚠ GPU check failed: {e}")
    
    # Check NVENC
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], 
                                capture_output=True, text=True, timeout=5)
        if "h264_nvenc" in result.stdout:
            print("✓ NVENC hardware encoding")
        else:
            print("⚠ NVENC not available - using software encoding")
    except Exception:
        pass
    
    print("=" * 50)
    
    if errors:
        print(f"FATAL: Missing required dependencies: {errors}")
        sys.exit(1)
    
    if warnings:
        print(f"Warnings: {warnings}")
    
    print("Starting application...")
    print("=" * 50 + "\n")
    
    # Import and run app
    from app import create_ui
    app = create_ui()
    app.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()
```

---

## 4. Backend Functions (EXACT Implementation Patterns)

### **VRAM Flush (REQUIRED at start of EVERY processing function)**
```python
def flush_vram():
    """Flush GPU memory - call at START of every processing function."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

### **WhisperX Transcription Pattern**
```python
def generate_subtitles(
    audio_file,  # gr.Audio(type="filepath") returns string path
    model_size: str,
    batch_size: int,
    language: str,
    hf_token: str,
    enable_diarization: bool,
    progress=gr.Progress()
):
    flush_vram()
    
    # Validate input
    if not audio_file or not Path(audio_file).exists():
        raise gr.Error("Please upload a valid audio/video file.")
    
    try:
        import whisperx
        
        progress(0.1, desc="Loading Whisper model...")
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        
        model = whisperx.load_model(
            model_size,
            device=DEVICE,
            compute_type=compute_type,
            download_root=str(WHISPER_DIR),
            language=language if language != "auto" else None
        )
        
        progress(0.2, desc="Loading audio...")
        audio = whisperx.load_audio(audio_file)
        
        progress(0.35, desc="Transcribing...")
        result = model.transcribe(audio, batch_size=batch_size)
        detected_lang = result["language"]
        
        # CRITICAL: Delete model to free VRAM before alignment
        del model
        flush_vram()
        
        progress(0.5, desc="Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=DEVICE
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, DEVICE,
            return_char_alignments=False
        )
        
        del model_a
        flush_vram()
        
        # Diarization (optional)
        if enable_diarization:
            if not hf_token:
                raise gr.Error(
                    "Speaker diarization requires a HuggingFace token.\n"
                    "1. Get token: huggingface.co/settings/tokens\n"
                    "2. Accept: huggingface.co/pyannote/segmentation-3.0\n"
                    "3. Accept: huggingface.co/pyannote/speaker-diarization-3.1"
                )
            progress(0.7, desc="Identifying speakers...")
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                del diarize_model
                flush_vram()
            except Exception as e:
                raise gr.Error(f"Diarization failed: {e}")
        
        progress(0.9, desc="Generating output files...")
        # ... generate SRT/ASS files ...
        
        progress(1.0, desc="Complete!")
        return transcript_text, srt_path, ass_path
        
    except gr.Error:
        raise
    except Exception as e:
        flush_vram()
        raise gr.Error(f"Transcription failed: {str(e)}")
```

### **Video Restoration Pattern (with proper frame handling)**
```python
def restore_video(video_file, target_res, upscale_model, tile_size, face_weight, enable_face, progress=gr.Progress()):
    flush_vram()
    
    if not video_file or not Path(video_file).exists():
        raise gr.Error("Please upload a video file.")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = TEMP_DIR / f"frames_{ts}"
    upscaled_dir = TEMP_DIR / f"upscaled_{ts}"
    
    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
        upscaled_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Probe video
        progress(0.05, desc="Analyzing video...")
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height", "-of", "json", video_file
        ], capture_output=True, text=True, check=True)
        info = json.loads(probe.stdout)["streams"][0]
        fps_num, fps_den = map(int, info["r_frame_rate"].split("/"))
        fps = fps_num / fps_den
        input_height = int(info["height"])
        
        # 2. Extract frames
        progress(0.1, desc="Extracting frames...")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_file, "-qscale:v", "1",
            str(frames_dir / "frame_%08d.png")
        ], capture_output=True, check=True)
        
        frame_count = len(list(frames_dir.glob("*.png")))
        if frame_count == 0:
            raise gr.Error("Failed to extract frames from video.")
        
        # 3. Calculate scale
        target_height = 2160 if target_res == "4K" else 1080
        scale = max(2, min(4, (target_height // input_height) or 2))
        
        # 4. Upscale
        progress(0.25, desc=f"Upscaling {frame_count} frames...")
        if not REALESRGAN_BIN.exists():
            raise gr.Error(f"Real-ESRGAN not found at {REALESRGAN_BIN}")
        
        result = subprocess.run([
            str(REALESRGAN_BIN), "-i", str(frames_dir), "-o", str(upscaled_dir),
            "-s", str(scale), "-n", upscale_model, "-t", str(tile_size), "-f", "png"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise gr.Error(f"Real-ESRGAN failed: {result.stderr}")
        
        # 5. Verify output
        upscaled_frames = sorted(upscaled_dir.glob("*.png"))
        if not upscaled_frames:
            raise gr.Error("Upscaling produced no output frames.")
        
        final_dir = upscaled_dir
        
        # 6. Face restoration (optional)
        if enable_face and face_weight > 0 and CODEFORMER_DIR.exists():
            progress(0.6, desc="Restoring faces...")
            # ... CodeFormer subprocess ...
        
        # 7. Encode video
        progress(0.8, desc="Encoding video...")
        output_path = OUTPUT_DIR / f"{Path(video_file).stem}_restored_{ts}.mp4"
        
        encode_cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(final_dir / "frame_%08d.png"),
            "-i", video_file, "-map", "0:v:0", "-map", "1:a:0?",
            "-c:a", "aac", "-b:a", "192k"
        ]
        
        if HAS_NVENC:
            encode_cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"])
        else:
            encode_cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
        
        encode_cmd.extend(["-movflags", "+faststart", str(output_path)])
        subprocess.run(encode_cmd, capture_output=True, check=True)
        
        progress(1.0, desc="Complete!")
        return str(output_path), str(output_path)
        
    except gr.Error:
        raise
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Processing failed: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        raise gr.Error(f"Video restoration failed: {str(e)}")
    finally:
        # ALWAYS clean up temp files
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(upscaled_dir, ignore_errors=True)
        flush_vram()
```

### **Subtitle Burning (CORRECT gr.File handling)**
```python
def burn_subtitles(video, subtitle_file, font_size, progress=gr.Progress()):
    flush_vram()
    
    if not video:
        raise gr.Error("Please upload a video.")
    if not subtitle_file:
        raise gr.Error("Please upload a subtitle file.")
    
    # CRITICAL: gr.File returns object with .name attribute for path
    sub_path = subtitle_file.name if hasattr(subtitle_file, 'name') else str(subtitle_file)
    
    if not Path(sub_path).exists():
        raise gr.Error("Subtitle file not found.")
    
    # ... rest of implementation with proper path escaping for ffmpeg ...
```

---

## 5. Gradio UI Requirements

### **Theme & Container**
```python
with gr.Blocks(
    title="Universal Media Studio",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate", 
        font=gr.themes.GoogleFont("Inter")
    ),
    css="""
        .gradio-container { max-width: 1400px !important; }
        footer { display: none !important; }
    """
) as app:
```

### **Tab 1: AI Subtitles**
- 2-column layout: controls left, results right
- `gr.Audio(type="filepath", sources=["upload", "microphone"])`
- `gr.Accordion("Advanced Settings", open=False)` for model/batch/diarization
- Big primary button: "Generate Subtitles"
- Show `progress=gr.Progress()` during processing

### **Tab 2: Visual Restoration**
- 2-column layout
- `gr.Video(sources=["upload"])`
- `gr.Radio(["1080p", "4K"])` for target resolution
- `gr.Accordion("Tweak Upscaler", open=False)` for tile size, face settings
- Video preview and download on right

### **Tab 3: Toolbox**
- Nested `gr.Tabs()` inside for: Burn Subtitles, Convert, Extract Audio
- Each sub-tab has 2-column layout
- For subtitle file upload: `gr.File(file_types=[".srt", ".ass", ".vtt"])`

### **CRITICAL Component Rules**
1. `gr.Audio(type="filepath")` → returns string path directly
2. `gr.Video` → returns string path directly  
3. `gr.File` → returns **object**, use `.name` for path
4. Always validate paths exist before processing
5. Use `gr.Error("message")` for user-visible errors

---

## 6. Dockerfile

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

# ... install system deps ...

# Install PyTorch FIRST with CUDA 12.8
RUN uv pip install --system torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# ... (rest of build) ...

ENV PATH="/app/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONPATH="/app"
```

---

## 7. requirements.txt

```text
# WhisperX stack
whisperx==3.7.4
pyannote.audio==3.3.2
ctranslate2>=4.5.0
faster-whisper>=1.1.1
nltk>=3.9.1
av<16.0.0
triton>=3.3.0; sys_platform == 'linux' and platform_machine == 'x86_64'

# Gradio
gradio==5.0.0

# Audio/Video Processing
demucs==4.0.1
deepfilternet==0.5.6
soundfile==0.12.1
librosa==0.10.2
opencv-python-headless==4.9.0.80

# Restored Utils
basicsr==1.4.2
facexlib==0.3.0
lpips==0.1.4
einops==0.7.0

# Core Utilities
numpy>=2.0.2
scipy>=1.12.0
Pillow>=10.2.0
tqdm>=4.66.2
requests>=2.31.0
huggingface_hub[hf_transfer]>=0.25.1
transformers>=4.48.0
hf_transfer>=0.1.5
```

---

## 8. download_models.py Key Fix

**CRITICAL: Use `compute_type="int8"` when downloading on CPU:**
```python
def download_whisper_model():
    import whisperx
    model = whisperx.load_model(
        "large-v3",
        device="cpu",
        compute_type="int8",  # NOT float16 on CPU!
        download_root=str(WHISPER_DIR)
    )
    del model
```

---

## 9. Deliverables Checklist

- [ ] `app.py` - All functions with proper error handling and VRAM management
- [ ] `start.py` - Validates deps before starting
- [ ] `download_models.py` - Fixed CPU compute type
- [ ] `requirements.txt` - Pinned versions
- [ ] `Dockerfile` - Production ready
- [ ] `.dockerignore` - Excludes temp files
- [ ] `README.md` - Clear setup instructions

**Test before deployment:**
```bash
python start.py  # Validates and starts
docker build -t universal-media-studio .
docker run --gpus all -p 7860:7860 universal-media-studio
```