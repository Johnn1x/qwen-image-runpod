FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Только LoRA
RUN mkdir -p /workspace/lora && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -L -o /workspace/lora/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
    https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors && \
    apt-get purge -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
