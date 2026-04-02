# qwen-image-runpod
.
Open-source RunPod serverless worker for the latest Qwen image model, `Qwen/Qwen-Image-2512`.

This repo exists because the common one-click templates are easy to misconfigure for large Hugging Face models. The failure mode is usually the one you hit: the model download spills onto the container filesystem and dies with `No space left on device` before the first request finishes.

This implementation fixes that by:

- forcing Hugging Face, Xet, Transformers, and temporary files onto container disk under `/workspace/model-storage`
- downloading the model snapshot into a stable directory on container disk before loading it
- keeping the extra Xet cache small so downloads do not silently double disk usage
- locking the model download path so multiple workers do not race on local storage
- failing early with a clear error if container disk free space is too small

## Container Image

This repository publishes a public container image to GitHub Container Registry via GitHub Actions.

- Registry: `ghcr.io/textcortex/qwen-image-runpod`
- Recommended tag style for deployment: immutable `sha-*` tags
- Convenience tag: `latest`

If you prefer the Docker registry flow in RunPod, use `Import from Docker Registry` and paste the image URL from the package page after the workflow finishes.

## Model

- Default model: `Qwen/Qwen-Image-2512`
- GPU target: 80 GB VRAM class GPUs
- Default response: base64-encoded PNG

You can override the model later with `QWEN_MODEL_ID`, but the repo is tuned for the current Qwen image release.

## Repo Layout

- `handler.py`: RunPod serverless handler and model bootstrap logic
- `Dockerfile`: container image for RunPod
- `runpod.toml`: sane default endpoint sizing
- `sample-request.json`: ready-to-send request body

## RunPod Deployment

### 1. Create a serverless endpoint from this repo

Recommended starting point:

- GPU: `A100 80GB` or `H100 80GB`
- Container disk: `150 GB`
- Execution timeout: `1800 seconds`
- Min workers: `1`
- Max workers: `1`

The default `runpod.toml` is intentionally conservative. Without a network volume, each brand-new worker has to download the model again, so keeping one worker warm is the safest default.

If you are deploying from Docker Registry instead of GitHub:

- Click `Import from Docker Registry`
- Use `ghcr.io/textcortex/qwen-image-runpod:latest` for convenience, or a `sha-*` tag for reproducibility
- Keep the same GPU, container disk, and timeout values listed above

### 2. First cold start

The first request downloads the model to container disk. That can take a while. Subsequent requests on the same warm worker reuse the downloaded snapshot.

This image also sets `RUNPOD_INIT_TIMEOUT=1800` so large cold starts have room to finish.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QWEN_MODEL_ID` | `Qwen/Qwen-Image-2512` | Hugging Face model ID |
| `MODEL_STORAGE_PATH` | `/workspace/model-storage` | Container-disk storage root for model and caches |
| `RUNPOD_VOLUME_PATH` | `/workspace/model-storage` | Backward-compatible alias for storage root |
| `MIN_STORAGE_FREE_GB` | `80` | Fail fast if free container disk space is too small |
| `HF_DOWNLOAD_MAX_WORKERS` | `4` | Limits parallel download pressure |
| `HF_XET_CHUNK_CACHE_SIZE_BYTES` | `0` | Disables the large extra Xet chunk cache |
| `HF_XET_SHARD_CACHE_SIZE_LIMIT` | `1073741824` | Caps shard cache at 1 GiB |
| `RUNPOD_INIT_TIMEOUT` | `1800` | Gives cold starts time to download the model |

## Request Format

Send a standard RunPod serverless payload:

```json
{
  "input": {
    "prompt": "A clean product shot of a silver laptop on a warm sandstone pedestal",
    "negative_prompt": "blurry, low resolution, extra fingers",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "seed": 42
  }
}
```

Supported input fields:

- `prompt` (required)
- `negative_prompt`
- `width`
- `height`
- `num_inference_steps`
- `true_cfg_scale`
- `cfg_scale` as an alias for `true_cfg_scale`
- `seed`
- `output_format` with `PNG` or `JPEG`

`width` and `height` must be positive multiples of `16`.

## Response Format

```json
{
  "image": "base64-encoded image bytes",
  "seed": 42,
  "model_id": "Qwen/Qwen-Image-2512",
  "output_format": "png",
  "latency_seconds": 28.41
}
```

## Warm-Up Request

The repo includes [`sample-request.json`](./sample-request.json). For the first deploy, send that request once and wait for the initial download to complete.

## Local Notes

If you want to run the code outside RunPod, point `MODEL_STORAGE_PATH` at a writable directory with enough free space.

Example:

```bash
mkdir -p /tmp/qwen-image-storage
MODEL_STORAGE_PATH=/tmp/qwen-image-storage python3 -u handler.py
```

That is only useful for syntax and startup checks unless you also have a suitable GPU.

## License

The code in this repo is licensed under Apache-2.0. The Qwen model license is separate and should be reviewed before production use.
