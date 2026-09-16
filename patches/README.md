# patches/ — vllm-omni MiniMax-H3 推理管线增强

本目录包含对 vllm-omni（MiniMax-H3 后端）推理服务的两组补丁。补丁只修改推理服务的
Python 胶水层，**不改动模型权重**，因此与官方权重完全兼容。

## 补丁内容

### 1. FL2VA 关键帧图生视频（首帧 / 尾帧 / 首尾帧）

- `packed_sequence.py` 底层本就定义了关键帧签名 `((0,), (-1,), (0,-1))`，
  但上游管线把 `keyframe_frame_indices` 硬编码为 `[0]`（仅首帧），且条件图只允许一张
- 补丁后：
  - `serving_video.py`：`task=fl2va` 时，`input_references` 上传的图片按内容解码为
    关键帧图列表（1–2 张），而非被当作参考视频丢弃
  - `pipeline_minimax_h3.py`：支持多张条件图（每张独立的 Qwen 视觉槽位与 VAE 条件行），
    从 `extra_params.frame_indices` 读取关键帧签名并校验
  - 客户端约定：`frame_indices=[0]`（首帧）、`[-1]`（尾帧）、`[0,-1]`（首尾帧，文件按首、尾顺序）

### 2. image-only Ref2VA（纯图片参考）

- 上游实现强制“图片参考必须搭配音频”（`image Ref2VA requires multi_modal_data.audio`）
- 官方输入矩阵（vllm-omni recipes/MiniMaxAI/MiniMax-H3.md）实际支持 image-only，
  仅 audio-only 被拒绝
- 补丁后：音频参考可选。无音频时不打包音频参考块、Qwen 表现层省略 `<Audio 1>` 槽位、
  去噪循环以 `audio_ref_rows=None` 运行（该分支上游原生支持）；有音频时行为与原版逐字节一致

## 应用方式

### 方式 A：Dockerfile（推荐，可复现）

`files/` 内含打好补丁的最终文件（基于本仓库冻结的 vllm-omni 版本）：

```bash
cd patches
docker build -t vllm/vllm-omni:minimax-h3 .
```

Dockerfile 以官方基础镜像为起点，仅覆盖两个 Python 文件。

### 方式 B：对运行中的容器在线应用

适合基础镜像版本与 `files/` 不同、需要按锚点即时重打的情况：

```bash
# 对正在运行的后端容器执行（默认容器名 minimax-h3-fl2va，可用 H3_PATCH_CONTAINER 覆盖）
python3 patch_backend.py            # 关键帧 FL2VA + serving 层
python3 patch_backend.py phase2     # 关键帧上传按内容解码（修正 .mp4 后缀问题）
python3 patch_backend.py phase3     # image-only Ref2VA

# 持久化为镜像并滚动重启
docker commit minimax-h3-fl2va vllm/vllm-omni:minimax-h3
docker compose up -d
```

补丁器特性：锚点唯一性校验（不匹配即中止，不会装出半成品）、容器内 `py_compile`
语法校验、原始文件双重备份（宿主机 `orig/` 与容器 `/tmp`）。

## 已验证

- `tests/test-first-last.sh`：尾帧、首尾帧两种模式均生成带音轨 MP4（ffprobe 校验）
- `tests/api_test.py`：29 项用例（含 sync/async、四种生成模式、鉴权、SSRF、持久化）全部通过
- image+audio Ref2VA 回归：补丁后与补丁前行为一致
