#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
case "${1:-}" in
  pull) docker compose pull ;;
  start|up) docker compose up -d ;;
  stop) docker compose stop ;;
  down) docker compose down ;;
  restart) docker compose restart ;;
  status) docker compose ps; nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv ;;
  logs) docker compose logs -f --tail="${2:-200}" ;;
  logs-fl2va) docker compose logs -f --tail="${2:-200}" h3-fl2va ;;
  logs-ref2va) docker compose logs -f --tail="${2:-200}" h3-ref2va ;;
  logs-gateway) docker compose logs -f --tail="${2:-200}" h3-gateway ;;
  key) sed -n 's/^H3_API_KEY=//p' .env ;;
  test-t2va) exec "$ROOT/test-t2va.sh" ;;
  test-fl2va) exec "$ROOT/test-fl2va.sh" ;;
  test-ref2va) exec "$ROOT/test-ref2va.sh" ;;
  test-gateway) exec "$ROOT/test-gateway-async.sh" ;;
  *) echo "Usage: $0 {pull|start|stop|down|restart|status|logs|logs-fl2va|logs-ref2va|logs-gateway|key|test-t2va|test-fl2va|test-ref2va|test-gateway}" >&2; exit 2 ;;
esac
