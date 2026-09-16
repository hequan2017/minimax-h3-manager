#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a
. ./.env
set +a
docker run --rm -v "$PWD/outputs:/outputs" vllm/vllm-omni:minimax-h3 ffmpeg -hide_banner -loglevel error -f lavfi -i "color=c=0xE67E22:s=448x256:d=1" -frames:v 1 -y /outputs/first-frame.png
docker run --rm -v "$PWD/outputs:/outputs" vllm/vllm-omni:minimax-h3 ffmpeg -hide_banner -loglevel error -f lavfi -i "color=c=0x27AE60:s=448x256:d=1" -frames:v 1 -y /outputs/last-frame.png
common=(-H "Authorization: Bearer $H3_API_KEY" "http://127.0.0.1:8090/v1/videos/sync" -F "model=MiniMax-H3-FL2VA" -F "width=448" -F "height=256" -F "fps=24" -F "num_inference_steps=2" -F "flow_shift=12")
curl -fsS --max-time 14400 "${common[@]}" -F "seed=71" -F "prompt=The scene naturally arrives at the provided final frame." --form-string 'extra_params={"task":"fl2va","duration":1.2,"frame_indices":[-1],"audio_flow_shift":3.0}' -F "input_references=@outputs/last-frame.png;type=image/png" -o outputs/tail-frame-smoke.mp4
curl -fsS --max-time 14400 "${common[@]}" -F "seed=72" -F "prompt=The scene transitions naturally from the first frame to the final frame." --form-string 'extra_params={"task":"fl2va","duration":1.2,"frame_indices":[0,-1],"audio_flow_shift":3.0}' -F "input_references=@outputs/first-frame.png;type=image/png" -F "input_references=@outputs/last-frame.png;type=image/png" -o outputs/first-last-smoke.mp4
for f in tail-frame-smoke.mp4 first-last-smoke.mp4; do echo "===$f==="; docker run --rm -v "$PWD/outputs:/outputs:ro" vllm/vllm-omni:minimax-h3 ffprobe -v error -show_entries format=duration,size:stream=codec_name,codec_type,width,height,sample_rate,channels -of compact=p=0:nk=1 "/outputs/$f"; done
