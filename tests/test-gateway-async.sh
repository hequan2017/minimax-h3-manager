#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a
. ./.env
set +a
response="$(curl -fsS -H "Authorization: Bearer $H3_API_KEY" http://127.0.0.1:8090/v1/videos   -F "model=MiniMax-H3-FL2VA"   -F "prompt=A small paper boat drifting across calm water with soft ambient sound."   -F "width=448" -F "height=256" -F "fps=24"   -F "num_inference_steps=2" -F "seed=61" -F "flow_shift=12"   --form-string 'extra_params={"task":"t2va","duration":1.2,"audio_flow_shift":3.0,"aspect_ratio":"16:9"}')"
echo "$response"
id="$(printf "%s" "$response" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
test -n "$id"
status=""
for i in $(seq 1 60); do
  state="$(curl -fsS -H "Authorization: Bearer $H3_API_KEY" "http://127.0.0.1:8090/v1/videos/$id")"
  status="$(printf "%s" "$state" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')"
  echo "poll=$i status=$status"
  test "$status" = "completed" && break
  if test "$status" = "failed"; then echo "$state"; exit 1; fi
  sleep 2
done
test "$status" = "completed"
curl -fsS -H "Authorization: Bearer $H3_API_KEY" "http://127.0.0.1:8090/v1/videos/$id/content" -o outputs/gateway-async.mp4
docker run --rm -v "$PWD/outputs:/outputs:ro" vllm/vllm-omni:minimax-h3 ffprobe -v error -show_entries format=duration,size:stream=codec_name,codec_type,width,height,sample_rate,channels -of compact=p=0:nk=1 /outputs/gateway-async.mp4
