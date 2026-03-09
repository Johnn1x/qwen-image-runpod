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


DEFAULT_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-2512")
DEFAULT_STORAGE_PATH = Path(
    os.getenv(
        "MODEL_STORAGE_PATH",
        os.getenv("RUNPOD_VOLUME_PATH", "/workspace/model-storage"),
    )
)
MIN_STORAGE_FREE_GB = int(os.getenv("MIN_STORAGE_FREE_GB", "80"))
HF_DOWNLOAD_MAX_WORKERS = int(os.getenv("HF_DOWNLOAD_MAX_WORKERS", "4"))


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

import runpod
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("qwen-image-runpod")

pipeline: DiffusionPipeline | None = None
pipeline_lock = Lock()


class UserInputError(ValueError):
    pass


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.1f} GiB"


def _disk_report(path: Path) -> str:
    total, used, free = shutil.disk_usage(path)
    return f"free={_format_gib(free)} total={_format_gib(total)} path={path}"


def _require_minimum_free_space(path: Path, min_free_gb: int) -> None:
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient free space on {path}. Found {free_gb:.1f} GiB free, "
            f"require at least {min_free_gb} GiB. Increase the RunPod container disk size."
        )


def _model_dir(model_id: str) -> Path:
    return MODEL_ROOT / model_id.replace("/", "--")


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        LOGGER.info("Waiting for model lock: %s", lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _download_model_snapshot(model_id: str) -> Path:
    local_dir = _model_dir(model_id)
    lock_path = LOCK_ROOT / f"{local_dir.name}.lock"
    force_sync = os.getenv("FORCE_MODEL_SYNC", "0") == "1"

    with _exclusive_lock(lock_path):
        ready_marker = local_dir / ".snapshot-complete"
        model_index = local_dir / "model_index.json"
        if ready_marker.exists() and model_index.exists() and not force_sync:
            LOGGER.info("Using cached model snapshot from %s", local_dir)
            return local_dir

        _require_minimum_free_space(STORAGE_ROOT, MIN_STORAGE_FREE_GB)
        LOGGER.info("Storage before download: %s", _disk_report(STORAGE_ROOT))
        LOGGER.info("Downloading %s into %s", model_id, local_dir)

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            max_workers=HF_DOWNLOAD_MAX_WORKERS,
            token=os.getenv("HF_TOKEN") or None,
        )

        if not model_index.exists():
            raise RuntimeError(
                f"Model download finished but {model_index} is missing. Check the storage directory contents."
            )

        ready_marker.touch()
        LOGGER.info("Storage after download: %s", _disk_report(STORAGE_ROOT))
        return local_dir


def _load_pipeline() -> DiffusionPipeline:
    global pipeline

    if pipeline is not None:
        return pipeline

    with pipeline_lock:
        if pipeline is not None:
            return pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for Qwen image generation.")

        model_dir = _download_model_snapshot(DEFAULT_MODEL_ID)
        torch_dtype = torch.bfloat16

        LOGGER.info("Loading pipeline from %s", model_dir)
        loaded_pipeline = DiffusionPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
            use_safetensors=True,
        )

        loaded_pipeline = loaded_pipeline.to("cuda")
        loaded_pipeline.set_progress_bar_config(disable=True)

        if getattr(loaded_pipeline, "vae", None) is not None:
            loaded_pipeline.vae.enable_tiling()

        pipeline = loaded_pipeline
        LOGGER.info("Model loaded on %s", torch.cuda.get_device_name(0))
        return pipeline


def _parse_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise UserInputError(f"{name} must be an integer") from exc

    if minimum is not None and parsed < minimum:
        raise UserInputError(f"{name} must be >= {minimum}")

    return parsed


def _parse_dimension(name: str, value: Any) -> int:
    parsed = _parse_int(name, value, minimum=16)
    if parsed % 16 != 0:
        raise UserInputError(f"{name} must be a positive multiple of 16")
    return parsed


def _parse_float(name: str, value: Any, *, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise UserInputError(f"{name} must be a number") from exc

    if minimum is not None and parsed < minimum:
        raise UserInputError(f"{name} must be >= {minimum}")

    return parsed


def _build_seed(value: Any) -> int:
    if value is None:
        return secrets.randbelow(2**31 - 1)
    return _parse_int("seed", value, minimum=0)


def _image_to_base64(image: Image.Image, output_format: str) -> str:
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if output_format == "JPEG":
        image = image.convert("RGB")
        save_kwargs["quality"] = 95

    image.save(buffer, format=output_format, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_image(job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_input = job.get("input") or {}
        if not isinstance(job_input, dict):
            raise UserInputError("input must be an object")

        prompt = str(job_input.get("prompt", "")).strip()
        if not prompt:
            raise UserInputError("prompt is required")

        negative_prompt = str(job_input.get("negative_prompt", "")).strip()
        width = _parse_dimension("width", job_input.get("width", 1024))
        height = _parse_dimension("height", job_input.get("height", 1024))
        num_inference_steps = _parse_int(
            "num_inference_steps",
            job_input.get("num_inference_steps", 40),
            minimum=1,
        )
        true_cfg_scale = _parse_float(
            "true_cfg_scale",
            job_input.get("true_cfg_scale", job_input.get("cfg_scale", 4.0)),
            minimum=0.0,
        )
        seed = _build_seed(job_input.get("seed"))
        output_format = str(job_input.get("output_format", "PNG")).upper()
        if output_format not in {"PNG", "JPEG"}:
            raise UserInputError("output_format must be PNG or JPEG")

        LOGGER.info(
            "Job %s prompt=%r width=%s height=%s steps=%s cfg=%s seed=%s",
            job.get("id", "unknown"),
            prompt[:120],
            width,
            height,
            num_inference_steps,
            true_cfg_scale,
            seed,
        )

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
    LOGGER.info("Storage status: %s", _disk_report(STORAGE_ROOT))
    runpod.serverless.start({"handler": generate_image})
