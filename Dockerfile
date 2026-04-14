FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Запекаем модель + LoRA один раз при сборке
RUN mkdir -p /models/qwen-image-edit-2511 /workspace/lora && \
    python -c '
from huggingface_hub import snapshot_download
print("Скачиваем Qwen/Qwen-Image-Edit-2511...")
snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2511",
    local_dir="/models/qwen-image-edit-2511",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "model_index.json"]
)
print("Модель скачана.")
' && \
    curl -L -o /workspace/lora/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
    https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
