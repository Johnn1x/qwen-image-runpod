from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import time
from threading import Lock
from typing import Any, Dict

import runpod

# ====================== НАСТРОЙКИ ======================
MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen/qwen-image-edit-2511")
LORA_PATH = "/workspace/lora"
LORA_WEIGHT = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
HF_TOKEN = os.getenv("HF_TOKEN")

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}

LOGGER = logging.getLogger("runpod")
logging.basicConfig(level=logging.INFO)
LOGGER.setLevel(logging.INFO)

pipeline = None
pipeline_lock = Lock()
inference_lock = Lock()   # если хочешь 100% safety


def _load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    with pipeline_lock:
        if pipeline is not None:
            return pipeline

        LOGGER.info("Loading model: %s", MODEL_ID)

        from diffusers import QwenImageEditPlusPipeline
        import torch

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=HF_TOKEN,
            low_cpu_mem_usage=True,
        )

        pipe.enable_attention_slicing()
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)

        LOGGER.info("Loading LoRA: %s/%s", LORA_PATH, LORA_WEIGHT)
        pipe.load_lora_weights(
            LORA_PATH,
            weight_name=LORA_WEIGHT,
            adapter_name="lightning",
        )
        pipe.set_adapters(["lightning"], adapter_weights=[1.0])

        if getattr(pipe, "vae", None) is not None:
            pipe.vae.enable_tiling()

        pipeline = pipe
        LOGGER.info("Model is ready.")
        return pipeline


def _base64_to_image(b64: str):
    if not isinstance(b64, str) or not b64:
        raise ValueError("Field 'image' must be a non-empty base64 string")

    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]

    from PIL import Image
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")


def _image_to_base64(image, fmt: str = "PNG", data_url: bool = False) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if data_url:
        mime = "jpeg" if fmt == "JPEG" else fmt.lower()
        return f"data:image/{mime};base64,{b64}"
    return b64


def _parse_dimension(name: str, value: Any) -> int:
    try:
        v = int(value)
    except Exception:
        raise ValueError(f"Invalid {name}: {value}")

    if v < 64 or v > 2048 or v % 8 != 0:
        raise ValueError(f"Invalid {name}: {v} (must be 64..2048 and divisible by 8)")
    return v


def _parse_int(name: str, value: Any, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        v = int(value)
    except Exception:
        raise ValueError(f"Invalid {name}: {value}")

    if v < minimum:
        v = minimum
    if maximum is not None and v > maximum:
        v = maximum
    return v


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Явный health check: когда input вообще отсутствует
        if job.get("input") is None:
            return {"status": "ready"}

        job_input = job["input"]

        # Валидация обязательных полей
        image_b64 = job_input.get("image")
        if not image_b64:
            raise ValueError("Missing required field: 'image'")

        prompt = str(job_input.get("prompt", "")).strip()
        negative_prompt = str(job_input.get("negative_prompt", "")).strip()

        width = _parse_dimension("width", job_input.get("width", 1024))
        height = _parse_dimension("height", job_input.get("height", 1024))
        num_inference_steps = _parse_int("num_inference_steps", job_input.get("num_inference_steps", 8), minimum=1, maximum=50)

        seed = int(job_input.get("seed", secrets.randbelow(2**32)))
        output_format = str(job_input.get("output_format", "PNG")).upper()
        if output_format not in ALLOWED_FORMATS:
            raise ValueError(f"Invalid output_format: {output_format}. Allowed: {sorted(ALLOWED_FORMATS)}")

        return_data_url = bool(job_input.get("return_data_url", False))

        image = _base64_to_image(image_b64)

        LOGGER.info("Job %s | steps=%d | %dx%d", job.get("id"), num_inference_steps, width, height)

        pipe = _load_pipeline()

        import torch
        generator = torch.Generator(device="cuda").manual_seed(seed)

        start = time.perf_counter()
        with torch.inference_mode():
            # Если хочешь гарантировать отсутствие гонок:
            with inference_lock:
                result = pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    # true_cfg_scale оставляй только если 100% уверен что этот аргумент поддерживается
                    # true_cfg_scale=1.0,
                )

        latency = time.perf_counter() - start
        encoded = _image_to_base64(result.images[0], fmt=output_format, data_url=return_data_url)

        return {
            "image": encoded,
            "seed": seed,
            "model_id": MODEL_ID,
            "output_format": output_format.lower(),
            "latency_seconds": round(latency, 2),
        }

    except ValueError as e:
        LOGGER.warning("Bad request: %s", e)
        return {"error": str(e), "error_type": "bad_request"}

    except Exception as e:
        LOGGER.exception("Internal error")
        return {"error": "internal_error", "details": str(e)}


if __name__ == "__main__":
    LOGGER.info("Worker starting... (low CPU/RAM loading, no sequential offload)")
    runpod.serverless.start({"handler": handler})
