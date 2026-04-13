FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QWEN_MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
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
    RUNPOD_INIT_TIMEOUT=3600 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST=8.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
