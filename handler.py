from __future__ import annotations
import base64
import io
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import torch
from diffusers import DiffusionPipeline
from PIL import Image

# ==================== НАСТРОЙКИ ====================
DEFAULT_BASE_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO_ID = os.getenv("LORA_REPO_ID", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors")

DEFAULT_STORAGE_PATH = Path(
    os.getenv("MODEL_STORAGE_PATH", os.getenv("RUNPOD_VOLUME_PATH", "/workspace/model-storage"))
)

# ==================== STORAGE SETUP (оставил как было) ====================
STORAGE_ROOT = DEFAULT_STORAGE_PATH
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("qwen-image-runpod")

pipeline: DiffusionPipeline | None = None
pipeline_lock = Lock()

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (все ваши, без изменений) ====================
def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.2f} GiB"

def _disk_report(path: Path) -> str:
    import shutil
    usage = shutil.disk_usage(path)
    return f"total={_format_gib(usage.total)} | used={_format_gib(usage.used)} | free={_format_gib(usage.free)}"

def _require_minimum_free_space(path: Path) -> None:
    import shutil
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < 80:
        raise RuntimeError(f"Недостаточно места: {free_gb:.1f} GiB свободно")

def _model_dir(model_id: str) -> Path:
    safe_name = model_id.replace("/", "__")
    return STORAGE_ROOT / "models" / safe_name

def _exclusive_lock(lock_path: Path):
    import fcntl
    lock_file = open(lock_path, "w")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file

def _download_model_snapshot(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    _require_minimum_free_space(STORAGE_ROOT)
    lock_path = STORAGE_ROOT / "locks" / f"{repo_id.replace('/', '__')}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    with _exclusive_lock(lock_path):
        if (local_dir / "model_index.json").exists():
            LOGGER.info(f"Модель {repo_id} уже есть")
            return
        LOGGER.info(f"Скачиваем {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )

def _load_pipeline() -> DiffusionPipeline:
    global pipeline
    with pipeline_lock:
        if pipeline is not None:
            return pipeline
        base_dir = _model_dir(DEFAULT_BASE_MODEL_ID)
        _download_model_snapshot(DEFAULT_BASE_MODEL_ID, base_dir)

        LOGGER.info("Загружаем DiffusionPipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            str(base_dir),
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        lora_dir = _model_dir(LORA_REPO_ID)
        _download_model_snapshot(LORA_REPO_ID, lora_dir)
        pipe.load_lora_weights(str(lora_dir), weight_name=LORA_WEIGHT_NAME)
        pipe.fuse_lora()
        pipe.to("cuda")
        pipeline = pipe
        LOGGER.info("Модель загружена")
        return pipeline

def _parse_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except:
        return default

def _parse_dimension(val: Any, default: int = 1024) -> int:
    v = _parse_int(val, default)
    return max(256, min(2048, v // 8 * 8))

def _parse_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except:
        return default

def _build_seed(seed: Any) -> int:
    if isinstance(seed, int) and 0 <= seed < 2**32:
        return seed
    return secrets.randbelow(2**32)

def _image_to_base64(image: Image.Image, output_format: str) -> str:
    buffer = io.BytesIO()
    if output_format == "JPEG":
        image = image.convert("RGB")
    image.save(buffer, format=output_format, quality=95 if output_format == "JPEG" else None)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ==================== LIFESPAN (упрощённый, без загрузки модели) ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("=== WORKER STARTED SUCCESSFULLY ===")
    LOGGER.info("Storage: %s", _disk_report(STORAGE_ROOT))
    yield
    LOGGER.info("Worker shutdown")

app = FastAPI(lifespan=lifespan)

@app.get("/ping")
async def ping():
    return {"status": "healthy"}   # всегда 200, т.к. модель грузится лениво

@app.post("/generate")
async def generate(request: Request):
    global pipeline
    job = await request.json()
    try:
        # Ленивая загрузка модели только при первом запросе
        with pipeline_lock:
            if pipeline is None:
                pipeline = _load_pipeline()

        # ==================== ВАШ ИНФЕРЕНС ====================
        job_input = job.get("input") or {}
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "")
        width = _parse_dimension(job_input.get("width"), 1024)
        height = _parse_dimension(job_input.get("height"), 1024)
        num_inference_steps = _parse_int(job_input.get("num_inference_steps"), 8)
        guidance_scale = _parse_float(job_input.get("guidance_scale"), 7.5)
        output_format = job_input.get("output_format", "PNG").upper()
        seed = _build_seed(job_input.get("seed"))

        start_time = time.time()
        generator = torch.Generator("cuda").manual_seed(seed)
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        encoded_image = _image_to_base64(result, output_format)
        latency = time.time() - start_time

        return {
            "image": encoded_image,
            "seed": seed,
            "model_id": f"{DEFAULT_BASE_MODEL_ID} + {LORA_WEIGHT_NAME}",
            "output_format": output_format.lower(),
            "latency_seconds": round(latency, 2),
        }
    except Exception as exc:
        LOGGER.error("Ошибка: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
