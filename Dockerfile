FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Скачиваем модель и LoRA
RUN mkdir -p /models/qwen-image-edit-2511 /workspace/lora

COPY download_model.py /workspace/download_model.py
RUN python3 /workspace/download_model.py

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
