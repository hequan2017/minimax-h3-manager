#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
AUDIO_B64="$(base64 -w0 outputs/ref-audio.mp3)"
AUDIO_REF="$(printf '{"audio_url":"data:audio/mpeg;base64,%s"}' "$AUDIO_B64")"
EXTRA='{"task":"ref2va","duration":1.2,"audio_flow_shift":3.0}'
set -a
. ./.env
set +a
curl -fsS --max-time 14400   -H "Authorization: Bearer $H3_API_KEY"   http://127.0.0.1:8092/v1/videos/sync   -F "model=MiniMax-H3-Ref2VA"   -F "prompt=The golden object rotates slowly in a cinematic blue scene, synchronized to the reference tone."   -F "width=448" -F "height=256" -F "fps=24"   -F "num_inference_steps=2" -F "seed=44" -F "flow_shift=12"   --form-string "extra_params=$EXTRA"   -F "input_reference=@outputs/fl2va-input.png;type=image/png"   --form-string "audio_reference=$AUDIO_REF"   -o outputs/ref2va-smoke.mp4
