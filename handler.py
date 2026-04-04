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

# ==================== НАСТРОЙКИ ====================
DEFAULT_BASE_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO_ID = os.getenv("LORA_REPO_ID", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors")

DEFAULT_STORAGE_PATH = Path(
    os.getenv("MODEL_STORAGE_PATH", os.getenv("RUNPOD_VOLUME_PATH", "/workspace/model-storage"))
)
MIN_STORAGE_FREE_GB = int(os.getenv("MIN_STORAGE_FREE_GB", "80"))
HF_DOWNLOAD_MAX_WORKERS = int(os.getenv("HF_DOWNLOAD_MAX_WORKERS", "4"))

# ==================== STORAGE SETUP ====================
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

# ==================== ЛОГИРОВАНИЕ ====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("qwen-image-runpod")

pipeline: DiffusionPipeline | None = None
pipeline_lock = Lock()

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.2f} GiB"

def _disk_report(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return (
        f"total={_format_gib(usage.total)} | "
        f"used={_format_gib(usage.used)} | "
        f"free={_format_gib(usage.free)}"
    )

def _require_minimum_free_space(path: Path) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < MIN_STORAGE_FREE_GB:
        raise RuntimeError(
            f"Недостаточно места на диске: {free_gb:.1f} GiB свободно, "
            f"требуется минимум {MIN_STORAGE_FREE_GB} GiB"
        )

def _model_dir(model_id: str) -> Path:
    safe_name = model_id.replace("/", "__")
    return MODEL_ROOT / safe_name

def _exclusive_lock(lock_path: Path):
    lock_file = open(lock_path, "w")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file

def _download_model_snapshot(repo_id: str, local_dir: Path) -> None:
    _require_minimum_free_space(STORAGE_ROOT)
    lock_path = LOCK_ROOT / f"{repo_id.replace('/', '__')}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    with _exclusive_lock(lock_path):
        if (local_dir / "model_index.json").exists():
            LOGGER.info(f"Модель {repo_id} уже скачана")
            return
        LOGGER.info(f"Скачиваем {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=HF_DOWNLOAD_MAX_WORKERS,
        )
        LOGGER.info(f"Скачивание {repo_id} завершено")

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

        # LoRA
        lora_dir = _model_dir(LORA_REPO_ID)
        _download_model_snapshot(LORA_REPO_ID, lora_dir)
        pipe.load_lora_weights(str(lora_dir), weight_name=LORA_WEIGHT_NAME)
        pipe.fuse_lora()

        pipe.to("cuda")
        pipeline = pipe
        LOGGER.info("Модель полностью загружена и готова")
        return pipeline

def _parse_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

def _parse_dimension(val: Any, default: int = 1024) -> int:
    v = _parse_int(val, default)
    return max(256, min(2048, v // 8 * 8))

def _parse_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _build_seed(seed: Any)
