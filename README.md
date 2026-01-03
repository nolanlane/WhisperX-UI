# 🎬 Universal Media Studio

A production-ready, AI-powered media processing application designed for RunPod deployment. Features a polished Gradio dashboard interface with robust backend handling for heavy AI models.

## Features

### 🎤 AI Subtitles (WhisperX)
- **Accurate transcription** with Whisper large-v3
- **Word-level timestamps** via forced alignment
- **Speaker diarization** (optional, requires HuggingFace token)
- **Export to SRT/ASS** formats
- Progress tracking with loading bar

### 🖼️ Visual Restoration
- **Video upscaling** to 1080p or 4K using Real-ESRGAN (ncnn-vulkan binary)
- **Face restoration** with CodeFormer
- **NVENC hardware encoding** when available
- Frame-by-frame processing with cleanup

### 🎧 Audio Tools
- **Noise reduction** with DeepFilterNet
- **Stem separation** with Demucs (vocals, drums, bass, other,etc)
- Memory-isolated subprocess execution

### 🛠️ FFmpeg Toolbox
- **Burn subtitles** into video
- **Convert** between formats (MP4, MKV, WebM, MOV, AVI)
- **Extract audio** to various formats (MP3, WAV, FLAC, AAC, Opus)
- Hardware-accelerated encoding with NVENC

## AI Stack

| Component | Implementation | Notes |
|-----------|---------------|-------|
| WhisperX | Python library | float16 (GPU) / int8 (CPU) |
| Demucs | subprocess | Memory isolation + GPU targeting |
| Real-ESRGAN | ncnn-vulkan binary | Vulkan hardware acceleration |
| CodeFormer | subprocess | Face restoration with fidelity control |
| DeepFilterNet | CLI (df-enhance) | Aggressive noise reduction (GPU) |
| FFmpeg | subprocess | Hardened path escaping & NVENC support |

## Technical Architecture

The application follows a **"Guardian Process"** pattern for maximum stability in containerized environments:

### Lifecycle Workflow
1.  **Orchestration (`start.py`)**: Bootstraps the environment, acquires runtime binaries (Real-ESRGAN/CodeFormer), and validates hardware acceleration (CUDA/Vulkan/NVENC).
2.  **Model Management (`download_models.py`)**: Ensures all AI weights are stored in the persistent storage volume (`/workspace/models` on RunPod).
3.  **Core Logic (`app.py`)**: A Gradio-based engine that orchestrates heavy media pipelines with aggressive VRAM management and process isolation.
4.  **Compatibility (`sitecustomize.py`)**: Injected via `PYTHONPATH` to apply global monkey patches (e.g., BasicSR/Torchvision fixes) across all child processes.

### Data Pipelines
-   **ASR**: `Audio -> WhisperX -> Alignment -> Diarization -> Export`
-   **Restoration**: `Video -> Frames (FFmpeg) -> NCNN (Binary) -> CodeFormer (PyTorch) -> Stitch (FFmpeg)`
-   **Audio Tools**: `Audio -> DeepFilterNet/Demucs -> Memory-Isolated Processing`

## Persistence & Performance

This application is optimized for **RunPod** and other containerized environments:

- **RunPod Auto-Persistence**: The app automatically detects if it's running on RunPod by checking for the `/workspace` mount. If present, it redirects all models, binaries, and outputs to the persistent volume automatically.
- **Path Mapping**:
    - **Models**: `/workspace/models`
    - **Binaries**: `/workspace/bin`
    - **Outputs**: `/workspace/outputs`
    - **Temp**: `/workspace/temp`
- **VRAM Optimization**: Aggressive memory flushing between AI stages ensures stability on 8GB+ GPUs.
- **Hardware Acceleration**: Automatically detects and utilizes **CUDA** (Torch), **Vulkan** (Real-ESRGAN), and **NVENC** (FFmpeg).
- **Auto-Cleanup**: Startup routine cleans up temporary files and outputs older than 24 hours to prevent disk exhaustion.

## Installation

### Local Development

```bash
# Clone the repository
git clone <your-repo>
cd WhisperX-UI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run the app
python app.py
```

### Docker Build

```bash
# Build the image locally
docker build -t whisperx-ui .

# Run the container
# Note: in the slim image configuration, models will download on first run.
docker run --gpus all -p 7860:7860 whisperx-ui
```

### GHCR (Recommended)

The Docker image is published automatically on every push to `master` via GitHub Actions.

**Image:**

```text
ghcr.io/nolanlane/whisperx-ui:latest
```

Pull and run:

```bash
docker pull ghcr.io/nolanlane/whisperx-ui:latest
docker run --gpus all -p 7860:7860 ghcr.io/nolanlane/whisperx-ui:latest
```

### RunPod Deployment

Use the published GHCR image:

1. Create a new RunPod template
2. Container image:
   - `ghcr.io/nolanlane/whisperx-ui:latest`
3. Set the exposed port to `7860`
4. Deploy

Notes:

- The container will download models on first startup if they are not already present in the persistent storage (`/workspace/models` on RunPod).
- For best experience, attach a persistent volume to `/workspace` so subsequent starts do not re-download models.

## Configuration

### Media Presets

The app includes intelligent presets that auto-configure settings:

| Preset | Best For | Upscale Model | Face Weight |
|--------|----------|---------------|-------------|
| Anime | Animation, cartoons | realesrgan-x4plus-anime | 0.3 |
| Live Action | Movies, TV shows | realesrgan-x4plus | 0.7 |
| Podcast | Audio-focused content | realesrgan-x4plus | 0.5 |

### Environment Variables

```bash
HF_HUB_ENABLE_HF_TRANSFER=1  # Faster model downloads
# Note: HF_HOME, NLTK_DATA, etc. are automatically handled and redirected to /workspace if available.
```

## Project Structure

```
WhisperX-UI/
├── app.py                 # Main Gradio application
├── start.py               # Startup validator + first-run model download
├── download_models.py     # Model downloader (used on first run in slim image)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── README.md             # This file
```

## Backend Safety Mechanisms

### VRAM Management
- `flush_vram()` runs `gc.collect()` and `torch.cuda.empty_cache()` at the start of every function
- Prevents OOM errors when switching between tabs/models

### Queueing
- App launches with `.queue()` enabled
- Prevents timeouts on long video renders
- Configurable concurrency limits

### Memory Isolation
- Demucs runs via subprocess to keep memory separate
- Real-ESRGAN uses pre-compiled binary (no Python VRAM overhead)

## UI Design

The interface follows UX best practices:

- **2-column layouts** for controls (left) and results (right)
- **Collapsible accordions** for advanced settings
- **Tabbed sub-interfaces** to reduce clutter
- **Progress indicators** for long-running tasks
- **Modern Soft theme** with Inter font

## Requirements

### Hardware
- NVIDIA GPU with CUDA 12.1+ support
- Minimum 8GB VRAM (16GB+ recommended for 4K processing)
- 32GB+ system RAM recommended

### Software
- Python 3.10+
- CUDA Toolkit 12.1
- FFmpeg with NVENC support (optional, for hardware encoding)

## Troubleshooting

### Out of Memory
- Reduce batch size in Advanced Settings
- Use smaller Whisper model (medium instead of large-v3)
- Reduce tile size for video upscaling

### Slow Processing
- Ensure GPU is being utilized (check `nvidia-smi`)
- Enable NVENC for faster video encoding
- Use SSD storage for temp files

### Diarization Not Working
1. Get a HuggingFace token from https://huggingface.co/settings/tokens
2. Accept model agreements:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Enter token in Advanced Settings

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) - Accurate speech recognition
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Image/video upscaling
- [CodeFormer](https://github.com/sczhou/CodeFormer) - Face restoration
- [Demucs](https://github.com/facebookresearch/demucs) - Audio source separation
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Noise suppression
- [Gradio](https://gradio.app/) - Web interface framework
