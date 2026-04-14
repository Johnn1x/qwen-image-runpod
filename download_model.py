from huggingface_hub import snapshot_download
import os

print("Скачиваем модель Qwen/Qwen-Image-Edit-2511...")
snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2511",
    local_dir="/models/qwen-image-edit-2511",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "model_index.json"]
)
print("Модель успешно скачана.")

print("Скачиваем LoRA...")
os.system(
    "curl -L -o /workspace/lora/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors "
    "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
)
print("LoRA скачана.")
