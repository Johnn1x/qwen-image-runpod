FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /workspace

# Установка базовых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# === Создаём виртуальное окружение (решает проблему с pip) ===
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем pip внутри venv
RUN pip install --upgrade pip setuptools wheel

# Установка PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Пути к модели
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QWEN_MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
    LORA_REPO_ID=lightx2v/Qwen-Image-Edit-2511-Lightning \
    LORA_WEIGHT_NAME=Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors \
    MODEL_STORAGE_PATH=/workspace/model-storage \
    RUNPOD_VOLUME_PATH=/workspace/model-storage \
    HF_HOME=/workspace/model-storage/huggingface \
    HF_HUB_CACHE=/workspace/model-storage/huggingface/hub \
    HF_ASSETS_CACHE=/workspace/model-storage/huggingface/assets \
    HF_XET_CACHE=/workspace/model-storage/huggingface/xet \
    TRANSFORMERS_CACHE=/workspace/model-storage/huggingface/transformers \
    TMPDIR=/workspace/model-storage/tmp \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    HF_XET_SHARD_CACHE_SIZE_LIMIT=1073741824 \
    HF_XET_NUM_CONCURRENT_RANGE_GETS=4 \
    MIN_STORAGE_FREE_GB=80 \
    HF_DOWNLOAD_MAX_WORKERS=4 \
    RUNPOD_INIT_TIMEOUT=1800

# Создаём директории для кэша
RUN mkdir -p /workspace/model-storage/huggingface /workspace/model-storage/tmp

COPY requirements.txt /workspace/requirements.txt

# Установка зависимостей
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
