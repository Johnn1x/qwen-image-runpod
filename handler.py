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
DEFAULT_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_PATH = "/workspace/lora"
LORA_WEIGHT = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"

LOGGER = logging.getLogger("runpod")
LOGGER.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

pipeline: QwenImageEditPlusPipeline | None = None
pipeline_lock = Lock()

# ====================== RUNPOD MODEL CACHE ======================
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Оффлайн-режим только на runtime
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_snapshot_path(model_id: str) -> str:
    if "/" not in model_id:
        raise ValueError(f"MODEL_ID '{model_id}' должен быть в формате 'org/name'")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if versions:
            versions.sort(reverse=True)
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(
        f"Модель не найдена в RunPod Model Cache: {model_id}\n"
        "Проверьте поле 'Model' в настройках Endpoint = Qwen/Qwen-Image-Edit-2511"
    )


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
        LOGGER.info("Модель + LoRA успешно загружены на %s", torch.cuda.get_device_name(0))
        return pipeline


# ====================== generate_image ======================
def _base64_to_image(b64: str) -> Image.Image:
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_dimension(name: str, value: Any) -> int:
    try:
        v = int(value)
        if v < 64 or v > 2048 or v % 8 != 0:
            raise ValueError
        return v
    except Exception:
        raise ValueError(f"Неверное значение {name}: {value}")


def _parse_int(name: str, value: Any, minimum: int = 1) -> int:
    try:
        v = int(value)
        return max(minimum, v)
    except Exception:
        return minimum


def generate_image(job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_input = job.get("input") or {}
        image_b64 = job_input.get("image")
        if not image_b64:
            LOGGER.info("Health check request received. Returning ready status.")
            return {"status": "ready"}

        image = _base64_to_image(image_b64)
        prompt = str(job_input.get("prompt", "")).strip()
        negative_prompt = str(job_input.get("negative_prompt", "")).strip()
        width = _parse_dimension("width", job_input.get("width", 1024))
        height = _parse_dimension("height", job_input.get("height", 1024))
        num_inference_steps = _parse_int("num_inference_steps", job_input.get("num_inference_steps", 8), minimum=1)
        seed = int(job_input.get("seed", secrets.randbelow(2**32)))
        output_format = str(job_input.get("output_format", "PNG")).upper()

        LOGGER.info("Job %s | steps=%s | size=%dx%d", job.get("id"), num_inference_steps, width, height)

        pipe = _load_pipeline()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=1.0,
                generator=generator,
                num_images_per_prompt=1,
            )

        latency_seconds = time.perf_counter() - start_time
        encoded_image = _image_to_base64(result.images[0], output_format)

        return {
            "image": encoded_image,
            "seed": seed,
            "model_id": DEFAULT_MODEL_ID,
            "output_format": output_format.lower(),
            "latency_seconds": round(latency_seconds, 2),
        }
    except Exception as exc:
        LOGGER.exception("Error in generate_image")
        return {"error": f"Internal error: {str(exc)}"}


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
