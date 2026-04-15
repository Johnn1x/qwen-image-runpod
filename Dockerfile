FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Минимум для скачивания модели (быстро)
RUN python3 -m pip install --no-cache-dir --upgrade pip huggingface_hub

# 2. Создаём папки
RUN mkdir -p /models/qwen-image-edit-2511 /workspace/lora

# 3. Копируем скрипт скачивания
COPY download_model.py /workspace/download_model.py

# 4. Временно разрешаем интернет только для скачивания
ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

RUN python3 -u /workspace/download_model.py

# 5. Сразу возвращаем offline-режим для runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 6. Устанавливаем все остальные зависимости (включая diffusers и т.д.)
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "handler.py"]
