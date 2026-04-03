FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /workspace

# Установка Python и необходимых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка PyTorch с CUDA 12.8
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Переменные окружения (исправленные пути)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QWEN_MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
    LORA_REPO_ID=lightx2v/Qwen-Image-Edit-2511-Lightning \
    LORA_WEIGHT_NAME=Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    MODEL_STORAGE_PATH=/runpod-volume/model-storage \
    HF_HOME=/runpod-volume/model-storage/huggingface \
    HF_HUB_CACHE=/runpod-volume/model-storage/huggingface/hub \
    HF_ASSETS_CACHE=/runpod-volume/model-storage/huggingface/assets \
    HF_XET_CACHE=/runpod-volume/model-storage/huggingface/xet \
    TRANSFORMERS_CACHE=/runpod-volume/model-storage/huggingface/transformers \
    TMPDIR=/runpod-volume/model-storage/tmp \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    HF_XET_SHARD_CACHE_SIZE_LIMIT=1073741824 \
    HF_XET_NUM_CONCURRENT_RANGE_GETS=4 \
    MIN_STORAGE_FREE_GB=80 \
    HF_DOWNLOAD_MAX_WORKERS=4 \
    RUNPOD_INIT_TIMEOUT=1800

# Создаём директории для кэша
RUN mkdir -p /runpod-volume/model-storage/huggingface /runpod-volume/model-storage/tmp

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
