from huggingface_hub import snapshot_download
import subprocess
import os
import sys

print("=== Начало скачивания модели ===", flush=True)
print(f"HF_TOKEN present: {bool(os.getenv('HF_TOKEN'))}", flush=True)

try:
    print("Скачиваем основную модель Qwen/Qwen-Image-Edit-2511...", flush=True)
    snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2511",
        local_dir="/models/qwen-image-edit-2511",
        local_dir_use_symlinks=False,
        resume_download=True,
        # allow_patterns убрали — скачиваем всё, чтобы не было ошибок
    )
    print("✅ Модель успешно скачана.", flush=True)
except Exception as e:
    print(f"❌ Ошибка при скачивании модели: {e}", flush=True)
    sys.exit(1)

try:
    print("Скачиваем LoRA...", flush=True)
    subprocess.run(
        [
            "curl", "-f", "-L", "-o", "/workspace/lora/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
            "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
        ],
        check=True,
        capture_output=True,
        text=True
    )
    print("✅ LoRA успешно скачана.", flush=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Ошибка при скачивании LoRA: {e.stderr}", flush=True)
    sys.exit(1)

print("=== Всё скачано успешно ===", flush=True)
