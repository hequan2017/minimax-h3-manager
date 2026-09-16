#!/usr/bin/env bash
# MiniMax H3 Manager 一键部署：
#   1. 前置检查（docker / compose / NVIDIA / 磁盘）
#   2. 生成 .env（含随机 API 密钥）
#   3. 从魔搭 ModelScope 下载 MiniMax-H3 模型权重（如缺失）
#   4. 构建含补丁的推理镜像（如缺失）
#   5. 启动三服务并等待 healthy
# 用法：bash deploy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------- 可通过环境变量覆盖的配置 ----------
H3_MODEL_DIR="${H3_MODEL_DIR:-/data2/MiniMax-H3}"
H3_MODELSCOPE_ID="${H3_MODELSCOPE_ID:-MiniMax/MiniMax-H3}"
H3_IMAGE="${H3_IMAGE:-vllm/vllm-omni:minimax-h3}"
H3_BASE_IMAGE="${H3_BASE_IMAGE:-vllm/vllm-omni:latest}"

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[deploy] 失败:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 1. 前置检查 ----------
command -v docker >/dev/null || fail "未检测到 docker，请先安装 Docker"
docker compose version >/dev/null 2>&1 || fail "未检测到 docker compose 插件"
command -v nvidia-smi >/dev/null || fail "未检测到 NVIDIA 驱动 (nvidia-smi)"
nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi 运行失败，请检查驱动/容器环境"
docker info 2>/dev/null | grep -qi nvidia || log "提示：docker info 未发现 nvidia runtime；若使用 CDI 方式请确认 GPU 对容器可用"
FREE_GB=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 300 ] || log "警告：当前磁盘剩余 ${FREE_GB}GB，模型+镜像建议预留 300GB 以上"
log "前置检查通过"

# ---------- 2. 生成 .env ----------
if [ ! -s .env ]; then
  if command -v openssl >/dev/null; then
    KEY="h3_$(openssl rand -hex 24)"
  else
    KEY="h3_$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fi
  {
    echo "# 由 deploy.sh 自动生成（$(date '+%F %T')）"
    echo "H3_API_KEY=$KEY"
    echo "H3_MODEL_DIR=$H3_MODEL_DIR"
    echo "H3_IMAGE=$H3_IMAGE"
  } > .env
  chmod 600 .env
  log "已生成 .env（密钥随机）"
else
  log "检测到已有 .env，跳过生成"
fi
set -a; . ./.env; set +a
H3_IMAGE="${H3_IMAGE:-vllm/vllm-omni:minimax-h3}"

# ---------- 3. 下载模型（魔搭 ModelScope） ----------
if [ -d "$H3_MODEL_DIR/FL2VA" ] && [ -d "$H3_MODEL_DIR/Ref2VA" ]; then
  log "模型目录已存在：$H3_MODEL_DIR（含 FL2VA/Ref2VA），跳过下载"
else
  log "开始从魔搭下载模型 $H3_MODELSCOPE_ID → $H3_MODEL_DIR（权重较大，可能需要很久）"
  if ! command -v modelscope >/dev/null; then
    log "安装 modelscope CLI（pip）"
    pip3 install -U "modelscope" || fail "modelscope 安装失败，可手动 pip3 install modelscope 后重试"
  fi
  mkdir -p "$H3_MODEL_DIR"
  modelscope download --model "$H3_MODELSCOPE_ID" --local_dir "$H3_MODEL_DIR" \
    || fail "模型下载失败；也可手动下载后放到 $H3_MODEL_DIR（需含 FL2VA/ 与 Ref2VA/ 子目录）"
  [ -d "$H3_MODEL_DIR/FL2VA" ] && [ -d "$H3_MODEL_DIR/Ref2VA" ] \
    || fail "下载完成但未找到 FL2VA/Ref2VA 子目录，请核对模型仓库结构"
fi

# ---------- 4. 构建推理镜像 ----------
if docker image inspect "$H3_IMAGE" >/dev/null 2>&1; then
  log "推理镜像已存在：$H3_IMAGE，跳过构建"
else
  log "构建推理镜像 $H3_IMAGE（基于 $H3_BASE_IMAGE + 补丁）"
  docker pull "$H3_BASE_IMAGE"
  docker build --build-arg BASE_IMAGE="$H3_BASE_IMAGE" -t "$H3_IMAGE" "$ROOT/patches" \
    || fail "镜像构建失败，请查看 patches/README.md"
fi

# ---------- 5. 启动服务 ----------
log "启动服务（首次加载模型需要数分钟）"
docker compose up -d
log "等待后端 healthy……"
for svc in h3-fl2va h3-ref2va; do
  until [ "$(docker inspect --format '{{.State.Health.Status}}' "$svc" 2>/dev/null)" = "healthy" ]; do
    sleep 15
    printf '.'
  done
  echo ""
  log "$svc 已就绪"
done

PORT="$(grep -E '0\.0\.0\.0:8090' compose.yaml | head -1 | sed 's/.*0\.0\.0\.0:\([0-9]*\).*/\1/')"
PORT="${PORT:-8090}"
log "部署完成 ✔"
echo "  控制台/文档 : http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/"
echo "  API 密钥    : ./manage.sh key"
echo "  健康检查    : curl http://127.0.0.1:${PORT}/health"
echo "  接口测试    : python3 tests/api_test.py"
