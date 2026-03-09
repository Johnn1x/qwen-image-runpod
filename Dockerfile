FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QWEN_MODEL_ID=Qwen/Qwen-Image-2512 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    HF_ASSETS_CACHE=/runpod-volume/huggingface/assets \
    HF_XET_CACHE=/runpod-volume/huggingface/xet \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/transformers \
    TMPDIR=/runpod-volume/tmp \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    HF_XET_SHARD_CACHE_SIZE_LIMIT=1073741824 \
    HF_XET_NUM_CONCURRENT_RANGE_GETS=4 \
    MIN_VOLUME_FREE_GB=100 \
    HF_DOWNLOAD_MAX_WORKERS=4 \
    RUNPOD_INIT_TIMEOUT=1800

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]

