# qwen-image-runpod

Open-source RunPod serverless worker for the latest Qwen image model, `Qwen/Qwen-Image-2512`.

This repo exists because the common one-click templates are easy to misconfigure for large Hugging Face models. The failure mode is usually the one you hit: the model download spills onto the container filesystem and dies with `No space left on device` before the first request finishes.

This implementation fixes that by:

- forcing Hugging Face, Xet, Transformers, and temporary files onto `/runpod-volume`
- downloading the model snapshot into a stable directory on the network volume before loading it
- keeping the extra Xet cache small so downloads do not silently double disk usage
- locking the model download path so multiple workers do not race on the shared volume
- failing early with a clear error if the volume is missing or too small

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

### 1. Create a network volume first

Use a network volume in the same region as the worker.

- Recommended size: `150 GB`
- Minimum practical free space target: `100 GB`

This repo assumes the volume is mounted at `/runpod-volume`.

### 2. Create a serverless endpoint from this repo

Recommended starting point:

- GPU: `A100 80GB` or `H100 80GB`
- Container disk: `20 GB`
- Network volume: `150 GB`
- Execution timeout: `1800 seconds`
- Max workers: `1`

The default `runpod.toml` is intentionally conservative. Once the shared model cache is warm and stable, you can increase worker count.

### 3. First cold start

The first request downloads the model to the network volume. That can take a while. After the cache is warm, subsequent cold starts should reuse the downloaded snapshot.

This image also sets `RUNPOD_INIT_TIMEOUT=1800` so large cold starts have room to finish.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QWEN_MODEL_ID` | `Qwen/Qwen-Image-2512` | Hugging Face model ID |
| `RUNPOD_VOLUME_PATH` | `/runpod-volume` | Mounted shared volume path |
| `MIN_VOLUME_FREE_GB` | `100` | Fail fast if the volume is too small |
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

If you want to run the code outside RunPod, point `RUNPOD_VOLUME_PATH` at a writable directory with enough free space.

Example:

```bash
mkdir -p /tmp/qwen-image-volume
RUNPOD_VOLUME_PATH=/tmp/qwen-image-volume python3 -u handler.py
```

That is only useful for syntax and startup checks unless you also have a suitable GPU.

## License

The code in this repo is licensed under Apache-2.0. The Qwen model license is separate and should be reviewed before production use.
