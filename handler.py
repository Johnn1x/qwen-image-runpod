from __future__ import annotations
import base64
import fcntl
import io
import logging
import os
import secrets
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

DEFAULT_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_WEIGHT = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# ... (весь код до _load_pipeline остаётся без изменений)

def _load_pipeline() -> DiffusionPipeline:
    global pipeline
    if pipeline is not None:
        return pipeline
    with pipeline_lock:
        if pipeline is not None:
            return pipeline

        model_dir = _download_model_snapshot(DEFAULT_MODEL_ID)
        torch_dtype = torch.bfloat16

        LOGGER.info("Loading base pipeline from %s", model_dir)
        loaded_pipeline = DiffusionPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
            use_safetensors=True,
        )
        loaded_pipeline = loaded_pipeline.to("cuda")
        loaded_pipeline.set_progress_bar_config(disable=True)

        # ← Загружаем вашу LoRA
        LOGGER.info("Loading Lightning LoRA: %s / %s", LORA_REPO, LORA_WEIGHT)
        loaded_pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT,
            adapter_name="lightning"
        )
        loaded_pipeline.set_adapters(["lightning"], adapter_weights=[1.0])

        if getattr(loaded_pipeline, "vae", None) is not None:
            loaded_pipeline.vae.enable_tiling()

        pipeline = loaded_pipeline
        LOGGER.info("Model + LoRA loaded on %s", torch.cuda.get_device_name(0))
        return pipeline

# ... (весь остальной код до generate_image остаётся)

def generate_image(job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_input = job.get("input") or {}
        prompt = str(job_input.get("prompt", "")).strip()
        negative_prompt = str(job_input.get("negative_prompt", "")).strip()
        width = _parse_dimension("width", job_input.get("width", 1024))
        height = _parse_dimension("height", job_input.get("height", 1024))
        num_inference_steps = _parse_int("num_inference_steps", job_input.get("num_inference_steps", 4), minimum=1)
        strength = _parse_float("strength", job_input.get("strength", 0.8), minimum=0.1)  # для минимального изменения окружения
        true_cfg_scale = 1.0  # для Lightning обязательно

        # ... (seed, output_format и т.д. как было)

        LOGGER.info("Job %s | steps=%s | strength=%.2f", job.get("id"), num_inference_steps, strength)

        pipe = _load_pipeline()
        start_time = time.perf_counter()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                strength=strength,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
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
    except UserInputError as exc:
        return {"error": str(exc)}

if __name__ == "__main__":
    LOGGER.info("Worker starting with storage root: %s", STORAGE_ROOT)
    runpod.serverless.start({"handler": generate_image})
