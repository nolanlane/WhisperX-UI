# =============================================================================
# Universal Media Studio - Dockerfile
# Optimized for RunPod with NVIDIA GPU support
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

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

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Application Setup
# =============================================================================
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# =============================================================================
# Clone CodeFormer Repository
# =============================================================================
RUN git clone https://github.com/sczhou/CodeFormer.git /app/CodeFormer \
    && cd /app/CodeFormer \
    && pip install -r requirements.txt \
    && python basicsr/setup.py develop

# =============================================================================
# Download realesrgan-ncnn-vulkan binary
# =============================================================================
RUN mkdir -p /app/bin \
    && cd /app/bin \
    && wget -q https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
    && unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
    && chmod +x realesrgan-ncnn-vulkan \
    && rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip

# Add binary to PATH
ENV PATH="/app/bin:${PATH}"

# =============================================================================
# Copy Application Files
# =============================================================================
COPY download_models.py start.py app.py ./

# Create necessary directories
RUN mkdir -p /app/models/whisper \
    /app/models/huggingface \
    /app/outputs \
    /app/temp \
    /app/uploads

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/app/models/huggingface
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
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

COPY --from=builder /app/CodeFormer /app/CodeFormer
COPY --from=builder /app/bin /app/bin
COPY --from=builder /app/download_models.py /app/download_models.py
COPY --from=builder /app/start.py /app/start.py
COPY --from=builder /app/app.py /app/app.py

ENV PATH="/app/bin:${PATH}"

RUN mkdir -p /app/models/whisper \
    /app/models/huggingface \
    /app/outputs \
    /app/temp \
    /app/uploads

# =============================================================================
# Expose Port and Set Entrypoint
# =============================================================================
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application via startup validator
CMD ["python", "start.py"]
