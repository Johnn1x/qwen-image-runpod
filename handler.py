from __future__ import annotations
import base64
import io
import logging
import os
import secrets
import time
from pathlib import Path
from threading import Lock
from typing import Any
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import runpod

# ====================== НАСТРОЙКИ ======================
DEFAULT_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHT = "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors"   # ← FP8 fused

STORAGE_ROOT = Path(os.getenv("MODEL_STORAGE_PATH", "/workspace/model-storage"))
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("runpod")
LOGGER.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

pipeline: QwenImageEditPlusPipeline | None = None
pipeline_lock = Lock()

def _download_model_snapshot(model_id: str) -> Path:
    from huggingface_hub import snapshot_download
    model_dir = STORAGE_ROOT / "models" / model_id.replace("/", "--")
    if not model_dir.exists():
        LOGGER.info("Downloading model %s...", model_id)
        snapshot_download(repo_id=model_id, local_dir=str(model_dir),
                          local_dir_use_symlinks=False, resume_download=True)
    return model_dir

def _load_pipeline() -> QwenImageEditPlusPipeline:
    global pipeline
    if pipeline is not None:
        return pipeline
    with pipeline_lock:
        if pipeline is not None:
            return pipeline

        model_dir = _download_model_snapshot(DEFAULT_MODEL_ID)
        LOGGER.info("Loading base pipeline from %s", model_dir)

        loaded_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float8_e4m3fn,          # ← важно для FP8
            local_files_only=True,
            use_safetensors=True,
        )

        loaded_pipeline = loaded_pipeline.to("cuda")
        loaded_pipeline.set_progress_bar_config(disable=True)

        LOGGER.info("Loading FP8 Lightning LoRA: %s / %s", LORA_REPO, LORA_WEIGHT)
        loaded_pipeline.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT, adapter_name="lightning")
        loaded_pipeline.set_adapters(["lightning"], adapter_weights=[1.0])

        if getattr(loaded_pipeline, "vae", None) is not None:
            loaded_pipeline.vae.enable_tiling()

        pipeline = loaded_pipeline
        LOGGER.info("Model + LoRA loaded on %s", torch.cuda.get_device_name(0))
        return pipeline

# ====================== generate_image (без изменений) ======================
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
        strength = _parse_float("strength", job_input.get("strength", 0.8), minimum=0.1)
        seed = int(job_input.get("seed", secrets.randbelow(2**32)))
        output_format = str(job_input.get("output_format", "PNG")).upper()

        LOGGER.info("Job %s | steps=%s | image=%dx%d", job.get("id"), num_inference_steps, image.width, image.height)

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
                strength=strength,
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

# (все вспомогательные функции _base64_to_image, _parse_dimension и т.д. оставь как были в предыдущей версии)

if __name__ == "__main__":
    LOGGER.info("Worker starting with storage root: %s", STORAGE_ROOT)
    LOGGER.info("Lazy loading + FP8 enabled.")
    runpod.serverless.start({"handler": generate_image})
