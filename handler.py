from __future__ import annotations
import base64
import io
import logging
import os
import secrets
import time
from threading import Lock
from typing import Any

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import runpod

# ====================== НАСТРОЙКИ ======================
DEFAULT_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen/qwen-image-edit-2511")
LORA_PATH = "/workspace/lora"
LORA_WEIGHT = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"   # ← важно: с большой Q

LOGGER = logging.getLogger("runpod")
LOGGER.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

pipeline: QwenImageEditPlusPipeline | None = None
pipeline_lock = Lock()

# ====================== RUNPOD MODEL CACHE ======================
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_snapshot_path(model_id: str) -> str:
    model_id = model_id.lower().strip()          # ← принудительно lowercase
    LOGGER.info("Ищем модель в кэше (lowercase): %s", model_id)

    if "/" not in model_id:
        raise ValueError(f"Неверный model_id: {model_id}")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    snapshots_dir = os.path.join(model_root, "snapshots")

    LOGGER.info("HF cache root exists = %s", os.path.isdir(HF_CACHE_ROOT))
    LOGGER.info("Model root exists = %s", os.path.isdir(model_root))
    LOGGER.info("Snapshots dir exists = %s", os.path.isdir(snapshots_dir))

    if os.path.isdir(snapshots_dir):
        versions = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if versions:
            versions.sort(reverse=True)
            snapshot_path = os.path.join(snapshots_dir, versions[0])
            LOGGER.info("✅ Модель найдена в кэше: %s", snapshot_path)
            return snapshot_path

    raise RuntimeError(f"Модель не найдена в RunPod Model Cache: {model_id}\nПроверьте поле 'Model' = qwen/qwen-image-edit-2511")


def _load_pipeline() -> QwenImageEditPlusPipeline:
    global pipeline
    if pipeline is not None:
        return pipeline

    with pipeline_lock:
        if pipeline is not None:
            return pipeline

        LOGGER.info("Загрузка модели из RunPod Model Cache: %s", DEFAULT_MODEL_ID)
        local_model_path = resolve_snapshot_path(DEFAULT_MODEL_ID)

        loaded_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
        )
        loaded_pipeline = loaded_pipeline.to("cuda")
        loaded_pipeline.set_progress_bar_config(disable=True)

        LOGGER.info("Загрузка Lightning LoRA из Docker-образа: %s", LORA_WEIGHT)
        loaded_pipeline.load_lora_weights(
            LORA_PATH,
            weight_name=LORA_WEIGHT,
            adapter_name="lightning"
        )
        loaded_pipeline.set_adapters(["lightning"], adapter_weights=[1.0])

        if getattr(loaded_pipeline, "vae", None) is not None:
            loaded_pipeline.vae.enable_tiling()

        pipeline = loaded_pipeline
        LOGGER.info("✅ Модель + LoRA успешно загружены на %s", torch.cuda.get_device_name(0))
        return pipeline


# ====================== generate_image (остальное без изменений) ======================
# ... (весь остальной код generate_image, _base64_to_image и т.д. оставь как у тебя сейчас)

if __name__ == "__main__":
    LOGGER.info("Worker starting...")
    try:
        LOGGER.info("Pre-loading model + LoRA (RunPod Model Cache mode)...")
        _load_pipeline()
        LOGGER.info("Model + LoRA pre-loaded successfully.")
    except Exception:
        LOGGER.exception("Preload failed (most likely cache is not ready yet). Worker will continue anyway.")

    LOGGER.info("Starting Serverless handler.")
    runpod.serverless.start({"handler": generate_image})
