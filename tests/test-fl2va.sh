#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; set -a; . ./.env; set +a
test -s outputs/fl2va-input.png || docker run --rm -v "$ROOT/outputs:/outputs" vllm/vllm-omni:minimax-h3 ffmpeg -hide_banner -loglevel error -f lavfi -i "color=c=0x2457A6:s=448x256:d=1" -frames:v 1 -y /outputs/fl2va-input.png
curl -fsS --max-time 14400 -H "Authorization: Bearer $H3_API_KEY" http://127.0.0.1:8091/v1/videos/sync   -F 'model=MiniMax-H3-FL2VA' -F 'prompt=The scene moves naturally with a gentle camera push and matching ambient sound.'   -F 'width=448' -F 'height=256' -F 'fps=24' -F 'num_inference_steps=2' -F 'seed=43' -F 'flow_shift=12'   --form-string 'extra_params={"task":"fl2va","duration":1.2,"audio_flow_shift":3.0}'   -F 'input_reference=@outputs/fl2va-input.png;type=image/png' -o outputs/fl2va-smoke.mp4
docker run --rm -v "$ROOT/outputs:/outputs:ro" vllm/vllm-omni:minimax-h3 ffprobe -v error -show_entries format=duration,size:stream=codec_name,codec_type,width,height,sample_rate,channels -of json /outputs/fl2va-smoke.mp4
