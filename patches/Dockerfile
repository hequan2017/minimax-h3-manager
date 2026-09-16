# 以官方 vllm/vllm-omni 镜像为基础，覆盖打过补丁的 MiniMax-H3 推理文件。
# 基础镜像标签请与你的 vllm-omni 版本对应（files/ 内补丁基于撰写时冻结的版本验证）。
ARG BASE_IMAGE=vllm/vllm-omni:latest
FROM ${BASE_IMAGE}

COPY files/serving_video.py /usr/local/lib/python3.12/dist-packages/vllm_omni/entrypoints/openai/serving_video.py
COPY files/pipeline_minimax_h3.py /usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py
