#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; set -a; . ./.env; set +a
curl -fsS --max-time 14400 -H "Authorization: Bearer $H3_API_KEY" http://127.0.0.1:8091/v1/videos/sync   -F 'model=MiniMax-H3-FL2VA' -F 'prompt=A calm cinematic sunrise over a mountain lake, gentle camera movement, natural ambient sound.'   -F 'width=448' -F 'height=256' -F 'fps=24' -F 'num_inference_steps=2' -F 'seed=42' -F 'flow_shift=12'   --form-string 'extra_params={"task":"t2va","duration":1.2,"aspect_ratio":"16:9","audio_flow_shift":3.0}'   -o outputs/t2va-smoke.mp4
docker run --rm -v "$ROOT/outputs:/outputs:ro" vllm/vllm-omni:minimax-h3 ffprobe -v error -show_entries format=duration,size:stream=codec_name,codec_type,width,height,sample_rate,channels -of json /outputs/t2va-smoke.mp4
