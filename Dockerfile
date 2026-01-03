# =============================================================================
# Universal Media Studio - Dockerfile
# Optimized for RunPod with NVIDIA GPU support
# =============================================================================

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python and CUDA environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# HuggingFace acceleration
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/app/models/huggingface
ENV NLTK_DATA=/app/models/nltk
ENV TORCH_HOME=/app/models/torch

# CUDA paths
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# =============================================================================
# System Dependencies (Builder)
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    git-lfs \
    wget \
    curl \
    aria2 \
    unzip \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    libsndfile1 \
    libsndfile1-dev \
    libsox-dev \
    sox \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libvulkan1 \
    libvulkan-dev \
    mesa-vulkan-drivers \
    ca-certificates \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install uv for faster builds
RUN pip install --upgrade pip setuptools wheel && \
    pip install uv

# =============================================================================
# Application Setup
# =============================================================================
WORKDIR /app

# Create isolated environments for conflicting audio tools
RUN python3 -m venv /opt/venv/deepfilternet && \
    /opt/venv/deepfilternet/bin/pip install --upgrade pip && \
    /opt/venv/deepfilternet/bin/pip install deepfilternet==0.5.6

RUN python3 -m venv /opt/venv/demucs && \
    /opt/venv/demucs/bin/pip install --upgrade pip && \
    /opt/venv/demucs/bin/pip install demucs==4.0.1

RUN python3 -m venv /opt/venv/codeformer && \
    /opt/venv/codeformer/bin/pip install --upgrade pip && \
    /opt/venv/codeformer/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 && \
    /opt/venv/codeformer/bin/pip install basicsr==1.4.2 facexlib==0.3.0 lpips==0.1.4 einops==0.7.0 opencv-python-headless

# Link isolated binaries to a known location
RUN mkdir -p /app/bin && \
    ln -s /opt/venv/deepfilternet/bin/df-enhance /app/bin/df-enhance && \
    ln -s /opt/venv/demucs/bin/demucs /app/bin/demucs

# Copy requirements first for layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support (using uv for speed)
# Updated to match WhisperX 3.7.4 requirements (torchaudio>=2.8.0)
RUN uv pip install --system torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining Python dependencies
RUN uv pip install --system -r requirements.txt

# =============================================================================
# Copy Application Files
# =============================================================================
COPY download_models.py start.py app.py sitecustomize.py ./

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/app/models/huggingface
ENV NLTK_DATA=/app/models/nltk
ENV TORCH_HOME=/app/models/torch
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget \
    unzip \
    ffmpeg \
    libsndfile1 \
    sox \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/bin /app/bin

COPY --from=builder /app/download_models.py /app/download_models.py
COPY --from=builder /app/start.py /app/start.py
COPY --from=builder /app/app.py /app/app.py
COPY --from=builder /app/sitecustomize.py /app/sitecustomize.py

ENV PATH="/app/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONPATH="/app"

RUN mkdir -p /app/models/whisper \
    /app/models/huggingface \
    /app/models/nltk \
    /app/models/torch \
    /app/outputs \
    /app/temp \
    /app/uploads

# =============================================================================
# Expose Port and Set Entrypoint
# =============================================================================
EXPOSE 7860

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application via startup validator
CMD ["python", "start.py"]
