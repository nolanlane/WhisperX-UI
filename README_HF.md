---
title: Universal Media Studio
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app_hf.py
pinned: false
license: mit
suggested_hardware: zero-a10g
suggested_storage: small
tags:
  - whisper
  - transcription
  - subtitles
  - audio
  - video
  - demucs
  - deepfilternet
---

# 🎬 Universal Media Studio

AI-powered media processing toolkit running on HuggingFace Spaces with ZeroGPU support.

## Features

### 🎤 AI Subtitles (WhisperX)
- **Transcription**: WhisperX with word-level timestamp alignment
- **Languages**: Auto-detect or specify from 15+ languages
- **Speaker Diarization**: Identify different speakers (requires HF token)
- **Export**: SRT and ASS subtitle formats

### 🎧 Audio Tools
- **DeepFilterNet**: AI-powered noise reduction
- **Demucs**: Stem separation (vocals, drums, bass, other)
- **Models**: htdemucs, htdemucs_ft, mdx_extra

### 🛠️ Toolbox (FFmpeg)
- **Burn Subtitles**: Hardcode SRT/ASS into video
- **Convert**: Transcode with codec options
- **Extract Audio**: MP3, WAV, FLAC, AAC, Opus

## Usage

### Speaker Diarization Setup
To use speaker diarization, you need a HuggingFace token with access to pyannote models:

1. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept the user agreement at:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Paste your token in the "HuggingFace Token" field

## Hardware

This Space uses **ZeroGPU** for GPU-accelerated inference:
- WhisperX transcription
- Demucs stem separation
- DeepFilterNet noise reduction

GPU is allocated on-demand when running these operations.

## Local Development

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/universal-media-studio

# Install dependencies
pip install -r requirements_hf.txt

# Run locally
python app_hf.py
```

## Tech Stack

| Component | Library |
|-----------|---------|
| Transcription | WhisperX (faster-whisper) |
| Alignment | WhisperX wav2vec2 |
| Diarization | pyannote.audio |
| Noise Reduction | DeepFilterNet |
| Stem Separation | Demucs |
| Video Processing | FFmpeg |
| UI | Gradio |

## Limitations

- **Video Upscaling**: Real-ESRGAN and CodeFormer are not available on HF Spaces (requires ncnn binary)
- **NVENC**: Hardware encoding not available, uses software encoding
- **File Size**: Limited by HF Spaces storage quotas
- **Processing Time**: Long videos may timeout on free tier

## Credits

- [WhisperX](https://github.com/m-bain/whisperX) - Timestamp-accurate transcription
- [Demucs](https://github.com/facebookresearch/demucs) - Music source separation
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Noise suppression
- [Gradio](https://gradio.app) - UI framework

## License

MIT License
