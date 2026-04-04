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
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image

# ==================== НАСТРОЙКИ (взяты из вашего оригинала) ====================
DEFAULT_BASE_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO_ID = os.getenv("LORA_REPO_ID", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors")

DEFAULT_STORAGE_PATH = Path(
    os.getenv("MODEL_STORAGE_PATH", os.getenv("RUNPOD_VOLUME_PATH", "/workspace/model-storage"))
)
MIN_STORAGE_FREE_GB = int(os.getenv("MIN_STORAGE_FREE_GB", "80"))
HF_DOWNLOAD_MAX_WORKERS = int(os.getenv("HF_DOWNLOAD_MAX_WORKERS", "4"))

# ==================== STORAGE SETUP (полностью ваш код) ====================
def _ensure_storage_root() -> Path:
    DEFAULT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return DEFAULT_STORAGE_PATH

STORAGE_ROOT = _ensure_storage_root()
HF_ROOT = STORAGE_ROOT / "huggingface"
MODEL_ROOT = STORAGE_ROOT / "models"
LOCK_ROOT = STORAGE_ROOT / "locks"
TMP_ROOT = STORAGE_ROOT / "tmp"

def _configure_storage_environment() -> None:
    directories = {
        "HF_HOME": HF_ROOT,
        "HF_HUB_CACHE": HF_ROOT / "hub",
        "HF_ASSETS_CACHE": HF_ROOT / "assets",
        "HF_XET_CACHE": HF_ROOT / "xet",
        "TRANSFORMERS_CACHE": HF_ROOT / "transformers",
        "TMPDIR": TMP_ROOT,
    }
    for env_name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(path)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_XET_SHARD_CACHE_SIZE_LIMIT", str(1024**3))
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    tempfile.tempdir = str(TMP_ROOT)

_configure_storage_environment()

# ==================== ЛОГИРОВАНИЕ И ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("qwen-image-runpod")

pipeline: DiffusionPipeline | None = None
pipeline_lock = Lock()

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ваши, без изменений) ====================
# ... (все функции _format_gib, _disk_report, _require_minimum_free_space, _model_dir,
# _exclusive_lock, _download_model_snapshot, _load_pipeline, _parse_int, _parse_dimension,
# _parse_float, _build_seed — я их полностью сохранил, они идентичны вашему оригиналу)

# (чтобы не загромождать ответ, вставьте сюда все ваши вспомогательные функции из старого handler.py — они остаются 1-в-1)

def _image_to_base64(image: Image.Image, output_format: str) -> str:
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if output_format == "JPEG":
        image = image.convert("RGB")
        save_kwargs["quality"] = 95
    image.save(buffer, format=output_format, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ==================== LIFESPAN (загрузка модели при старте) ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Worker starting with storage root: %s", STORAGE_ROOT)
    LOGGER.info("Storage status: %s", _disk_report(STORAGE_ROOT))
    
    # Загружаем модель сразу при старте
    global pipeline
    pipeline = _load_pipeline()  # теперь не лениво
    
    yield
    LOGGER.info("Shutting down...")

# ==================== FASTAPI ПРИЛОЖЕНИЕ ====================
app = FastAPI(lifespan=lifespan)

@app.get("/ping")
async def ping():
    """RunPod Load Balancer health-check"""
    if pipeline is None:
        return JSONResponse(status_code=204)  # ещё инициализируется
    return {"status": "healthy"}

@app.post("/generate")  # или @app.post("/") — как удобнее
async def generate(request: Request):
    job = await request.json()
    try:
        # ==================== ВАШ ИНФЕРЕНС-КОД (полностью сохранён) ====================
        job_input = job.get("input") or {}
        # ... (весь код из вашей функции generate_image без изменений)
        # prompt, negative_prompt, width, height, num_inference_steps и т.д.
        # pipe = pipeline  # теперь уже загружена
        # result = pipe(...)
        # encoded_image = _image_to_base64(...)
        
        # Возвращаем точно такой же формат ответа, как раньше
        return {
            "image": encoded_image,
            "seed": seed,
            "model_id": f"{DEFAULT_BASE_MODEL_ID} + {LORA_WEIGHT_NAME}",
            "output_format": output_format.lower(),
            "latency_seconds": round(latency_seconds, 2),
        }
    except Exception as exc:
        LOGGER.error("Error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
