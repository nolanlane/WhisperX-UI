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
- **Stem separation** with Demucs (vocals, drums, bass, other)
- Memory-isolated subprocess execution

### 🛠️ FFmpeg Toolbox
- **Burn subtitles** into video
- **Convert** between formats (MP4, MKV, WebM, MOV, AVI)
- **Extract audio** to various formats (MP3, WAV, FLAC, AAC, Opus)
- Hardware-accelerated encoding with NVENC

## AI Stack

| Component | Implementation | Notes |
|-----------|---------------|-------|
| WhisperX | Python library | float16 compute |
| Demucs | subprocess | Memory isolation |
| Real-ESRGAN | ncnn-vulkan binary | No Python deps |
| CodeFormer | subprocess | inference_codeformer.py |
| DeepFilterNet | CLI (deepFilter) | Full-band audio |
| FFmpeg | ffmpeg-python + subprocess | NVENC when available |

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
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run the app
python app.py
```

### Docker Build

```bash
# Build the image (downloads all models at build time)
docker build -t universal-media-studio .

# Run the container
docker run --gpus all -p 7860:7860 universal-media-studio
```

### RunPod Deployment

1. Create a new RunPod template with this Dockerfile
2. Set the exposed port to `7860`
3. Deploy - models are pre-cached for instant startup

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
HF_HOME=/app/models/huggingface  # Model cache location
```

## Project Structure

```
WhisperX-UI/
├── app.py                 # Main Gradio application
├── download_models.py     # Build-time model downloader
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
