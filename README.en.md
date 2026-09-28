[简体中文](README.md) | [English](README.en.md)

# MiniMax H3 Manager

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](patches/Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-gateway-teal)](gateway/proxy.py)
[![vllm-omni](https://img.shields.io/badge/Inference-vllm--omni-orange)](patches/)

A **unified MiniMax-H3 video generation service** for self-hosted GPU servers: one Docker Compose stack (dual inference backends + a unified gateway), a Chinese web console with interactive API documentation, plus a set of enhancement patches for the vllm-omni inference pipeline.

> MiniMax-H3 is a 33B omni-modal (joint video+audio) generation model that outputs 4–15 second, 24FPS videos with a native stereo audio track.
> This project does not include model weights — it provides service orchestration, an API gateway and inference enhancements only.

Callers see a single model name, `MiniMax-H3`: the gateway routes each request to the FL2VA (text-to-video / image-to-video) or Ref2VA (reference generation) backend based on its content, so a single Video Generation API-style interface covers "create → poll → download".

## ✨ Features

### Unified API Gateway (port 8090)

- **Unified model name**: only `MiniMax-H3` is exposed; the gateway routes to the FL2VA / Ref2VA backend automatically per request
- **Async job API**: `POST /v1/videos` create → poll → download, compatible with the common Video Generation API style
- **Synchronous API**: `POST /v1/videos/sync` returns the MP4 directly — handy for scripts and tests
- **Model listing**: `GET /v1/models` returns the unified model name for OpenAI-style clients
- **Bearer key auth**: the key lives in the server-side `.env` (mode 600) and is validated and passed through to the backends
- **Persistent job history**: SQLite file database + videos on disk, so **history and videos remain playable after gateway restarts**
- **SSRF protection**: reference URLs are restricted to data URLs and public http(s); private / loopback / reserved / multicast addresses are rejected
- **Job metadata**: history keeps the task type, prompt and generation parameters for traceability

### Supported Generation Modes

| Mode | Input | Notes |
|------|------|------|
| Text-to-video (t2va) | Prompt | Aspect ratio is required |
| First-frame image-to-video | Prompt + first frame | Extends naturally from the given first frame |
| Last-frame image-to-video | Prompt + last frame | Motion settles onto the given last frame |
| First-last frame image-to-video | Prompt + first + last frame | Smooth interpolation between the two frames |
| Image reference generation | Prompt + reference images (1–4) | Multi-subject consistency; **audio optional**, used to specify the voice |
| Video reference generation | Prompt + reference video | The source video's audio track drives motion and sound |

Resolutions: 480P / 720P (768P). Ratios: 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16. Duration: 4–15 seconds.
Aspect ratio notes: text-to-video, image-reference and video-reference render at the chosen ratio; for first/last-frame image-to-video the frame is determined by the first frame image (pass `adaptive`).

### Web Console (Chinese, interactive docs)

- Form-based creation for every task type; wait for generation and play the result on the same page
- History page: task type / parameters / prompt / elapsed time at a glance; in-progress jobs get a **live wait timer + 5-second auto refresh**, with cancel and delete
- The API key can be stored in browser localStorage (current browser only, never uploaded)
- Full Chinese API documentation with curl examples built in

### Inference Pipeline Patches (`patches/`)

Upstream vllm-omni's MiniMax-H3 support matrix is narrow; the patches in this repo unlock the following without touching model weights:

1. **First / last / first-last frame image-to-video**: enables the `(0,) / (-1,) / (0,-1)`
   keyframe signatures from `MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES` (upstream glue hardcodes the first frame);
   multi-image conditioning, multi-slot vision embeddings and frame-index packing all work end to end
2. **image-only Ref2VA**: lifts the legacy "image reference requires audio" restriction so pure image references work
   (matching the official input matrix: only audio-only is rejected)

Patches are applied via "anchor replacement + in-container syntax check + abort on failure" (see `patches/patch_backend.py`);
`patches/files/` contains the final patched files, and an image can also be built directly with the Dockerfile.

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Gateway | Python 3.12 + FastAPI + httpx (`gateway/proxy.py`, single file) |
| Inference | vllm-omni (FL2VA / Ref2VA dual backends, 4+2 GPU tensor parallelism) |
| Console | Vanilla HTML/JS single page (`gateway/index.html`, Chinese) |
| Storage | SQLite (job metadata) + local MP4 files (`gateway_data/`) |
| Deployment | Docker Compose, three services: `h3-fl2va`(GPU 0–3) / `h3-ref2va`(GPU 4–5) / `h3-gateway`(:8090) |

## 🚀 Quick Start

### 0. Prerequisites

- NVIDIA GPU server (default allocation: GPU 0–3 for FL2VA, GPU 4–5 for Ref2VA; adjustable in compose.yaml)
- Docker + NVIDIA Container Toolkit + docker compose
- ≥300GB free disk (models + images)

### One-Click Deploy (recommended)

```bash
bash deploy.sh
```

The script runs: pre-checks → generates `.env` (random API key) → downloads the MiniMax-H3
weights from ModelScope (skipped if present; override the repo ID with `H3_MODELSCOPE_ID`) → builds the
patched inference image → starts the three services and waits until healthy, then prints the console URL and key location.

### Manual Deploy

#### 1. Build the inference image

```bash
# Option A: Dockerfile (recommended)
cd patches
docker build -t vllm/vllm-omni:minimax-h3 .

# Option B: apply patches to a running container, then commit
python3 patches/patch_backend.py            # run inside the started base container
docker commit minimax-h3-fl2va vllm/vllm-omni:minimax-h3
```

### 2. Configure and Start

```bash
cp .env.example .env
# Edit .env: set H3_API_KEY (a custom random string) and H3_MODEL_DIR (weights directory)
./manage.sh start
./manage.sh status          # wait for both backends to go healthy (first load takes minutes)
```

Environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `H3_API_KEY` | none (`deploy.sh` can generate one) | Public Bearer key for the gateway |
| `H3_MODEL_DIR` | `/data2/MiniMax-H3` | Weights directory, must contain `FL2VA/` and `Ref2VA/` subdirectories |
| `H3_IMAGE` | `vllm/vllm-omni:minimax-h3` | Patched inference image |
| `H3_MODELSCOPE_ID` | `MiniMax/MiniMax-H3` | ModelScope repo ID (used by `deploy.sh` downloads only) |

### 3. Usage

- Open `http://<server-ip>:8090` in a browser — create jobs, browse history, read the docs
- Call the API:

```bash
curl -X POST "http://localhost:8090/v1/videos" \
  -H "Authorization: Bearer $H3_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-H3",
    "content": [{"type": "text", "text": "一只橘猫在草原上奔跑。"}],
    "resolution": "768P",
    "duration": 5,
    "ratio": "16:9"
  }'
```

### 4. Testing

```bash
python3 tests/api_test.py   # 29 end-to-end API tests, writes a Markdown report
./manage.sh key             # show the server-side key
```

`tests/` also ships per-mode smoke scripts: `test-t2va.sh` / `test-fl2va.sh` / `test-ref2va.sh` / `test-gateway-async.sh`.

## 📁 Directory Layout

```
minimax-h3-manager/
├── compose.yaml            # three-service stack: h3-fl2va(GPU0-3) / h3-ref2va(GPU4-5) / h3-gateway
├── manage.sh               # one script for start/stop / logs / status / key
├── gateway/
│   ├── proxy.py            # unified gateway (FastAPI): routing, auth, async jobs, persistence, SSRF guard
│   └── index.html          # Chinese web console + interactive docs
├── patches/
│   ├── patch_backend.py    # patch applier (anchor-based, aborts on failure)
│   ├── Dockerfile          # builds directly from the official image + patched files
│   ├── files/              # final patched vllm-omni sources (COPY-ed by the Dockerfile)
│   └── README.md           # patch notes and application steps
├── tests/
│   ├── api_test.py         # full API test suite (29 cases incl. persistence checks), Markdown report
│   └── test-*.sh           # per-mode smoke scripts
├── docs/                   # showcase page and screenshots (GitHub Pages friendly)
├── .env.example            # environment variable sample
└── .gitignore
```

## 📸 Screenshots

![Create page](docs/img/screenshot-create.png)

![History page](docs/img/screenshot-history.png)

## 🔗 Related Projects

- [new-aigc-comfyui-minimax-h3](https://github.com/hequan2017/new-aigc-comfyui-minimax-h3) (ComfyStudio) — the same author's ComfyUI multi-GPU management platform: Go + Vue3 managing 8×L40 compute nodes, with built-in MiniMax H3 workflows and an AI comic-drama workbench. Complementary to this project's vllm-omni inference path.

## API Summary

| Method | Path | Description |
|------|------|------|
| POST | `/v1/videos` | Create an async job (202 returns the job ID) |
| GET | `/v1/videos` | Job list (with task type / prompt / parameters) |
| GET | `/v1/videos/{id}` | Query status: queued / in_progress / completed / failed / cancelled |
| GET | `/v1/videos/{id}/content` | Download the MP4 |
| DELETE | `/v1/videos/{id}` | Cancel a running job / delete the record and files |
| POST | `/v1/videos/sync` | Synchronous generation, returns the MP4 directly |
| GET | `/v1/models` | Model list (unified name `MiniMax-H3`) |
| GET | `/health` | Aggregated health check of both backends |

The request body centers on a `content[]` multimodal array: a `text` prompt plus media items with `role`
(`first_frame` / `last_frame` / `reference_image` / `reference_audio` / `reference_video`);
see the built-in documentation on the gateway home page for full field details.

## ⚠️ Known Limitations

- The vllm-omni inference path's reference matrix is 1 image (+1 audio) or multi-segment video; wider official multi-image combinations require the ComfyUI path (see related projects)
- 2K resolution is not enabled on this L40 setup; inference defaults to 50 steps (quality first) with no distillation acceleration
- The gateway key travels over plaintext HTTP — expose port 8090 only on trusted networks
- Pure image-reference generation (image-only Ref2VA) relies on this repo's patches; the generated audio track is left to the model

## 📄 License

[Apache-2.0](LICENSE). Model weights follow MiniMax's official license — confirm commercial terms separately.
