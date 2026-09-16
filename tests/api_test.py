#!/usr/bin/env python3
"""Full API test suite for the MiniMax H3 unified gateway.

Run on the GPU host:  python3 tests/api_test.py
- Reads H3_API_KEY from the project .env (never hardcoded).
- Exercises every public endpoint, both keyframe modes, persistence across a
  gateway restart, and error/SSRF paths.
- Every request targets a literal local connection (127.0.0.1:8090) with a
  fixed path allowlist — no dynamic hosts, no redirects followed.
- Writes a markdown report to tests/api_test_report.md.
"""

import base64
import http.client
import json
import os
import pathlib
import subprocess
import time

ALLOWED_PATH_PREFIXES = ("/", "/health", "/v1/models", "/v1/videos")
HOST = "127.0.0.1"
PORT = 8090
ROOT = pathlib.Path(os.environ.get("H3_PROJECT_ROOT", pathlib.Path(__file__).resolve().parent.parent))
OUT = ROOT / "outputs"
REPORT = ROOT / "tests" / "api_test_report.md"
POLL_TIMEOUT = 900
RESULTS = []


def load_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("H3_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("H3_API_KEY not found in .env")


KEY = load_key()


def request(method: str, path: str, *, key: str = KEY, body=None, content_type="application/json", timeout=14400):
    if not isinstance(path, str) or not path.startswith(ALLOWED_PATH_PREFIXES):
        raise ValueError(f"test path outside allowlist: {path!r}")
    headers = {}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = content_type
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        payload = resp.read()
        return resp.status, {k: v for k, v in resp.getheaders()}, payload
    finally:
        conn.close()


def data_url(path: pathlib.Path, mime: str) -> str:
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()


def record(name: str, ok: bool, detail: str):
    RESULTS.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + " — " + detail, flush=True)


def is_mp4(payload: bytes) -> bool:
    return len(payload) > 20000 and payload[4:8] == b"ftyp"


IMAGE_TAG = os.environ.get("H3_IMAGE", "vllm/vllm-omni:minimax-h3")


def ensure_fixtures() -> None:
    """Generate the small test media files into outputs/ if missing."""
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ("first-frame.png", "color=c=0xE67E22:s=448x256:d=1"),
        ("last-frame.png", "color=c=0x27AE60:s=448x256:d=1"),
        ("fl2va-input.png", "color=c=0x2457A6:s=448x256:d=1"),
    ]
    for name, lavfi in fixtures:
        target = OUT / name
        if target.is_file():
            continue
        subprocess.run(
            ["docker", "run", "--rm", "-v", str(OUT) + ":/outputs",
             IMAGE_TAG, "ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", lavfi, "-frames:v", "1", "-y", "/outputs/" + name],
            check=True,
            shell=False,
        )
    audio = OUT / "ref-audio.mp3"
    if not audio.is_file():
        subprocess.run(
            ["docker", "run", "--rm", "-v", str(OUT) + ":/outputs",
             IMAGE_TAG, "ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-q:a", "9", "-y", "/outputs/ref-audio.mp3"],
            check=True,
            shell=False,
        )


def wait_job(job_id: str):
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        status, _, payload = request("GET", "/v1/videos/" + job_id)
        if status != 200:
            return {"status": "http_" + str(status), "raw": payload[:300]}
        job = json.loads(payload)
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(5)
    return {"status": "poll_timeout"}


def main() -> None:
    ensure_fixtures()
    # -- static endpoints ---------------------------------------------------
    status, headers, payload = request("GET", "/", key="")
    record("GET / 文档页", status == 200 and b"MiniMax H3" in payload, f"HTTP {status}, {len(payload)} bytes")

    status, _, payload = request("GET", "/health", key="")
    health = json.loads(payload) if status == 200 else {}
    ok = status == 200 and all(s.get("ok") for s in health.get("services", {}).values())
    record("GET /health", ok, f"HTTP {status} {json.dumps(health.get('services'))}")

    status, _, _ = request("GET", "/v1/models", key="")
    record("GET /v1/models 无密钥 → 401", status == 401, f"HTTP {status}")
    status, _, _ = request("GET", "/v1/models", key="Bearer wrong-key-123")
    record("GET /v1/models 错误密钥 → 401", status == 401, f"HTTP {status}")
    status, _, payload = request("GET", "/v1/models")
    ok = status == 200 and "MiniMax-H3" in payload.decode()
    record("GET /v1/models 有效密钥", ok, f"HTTP {status}")

    status, _, payload = request("POST", "/v1/videos", body=b"not json", content_type="text/plain", key="")
    record("POST /v1/videos 非 JSON → 415", status == 415, f"HTTP {status}")

    # -- validation errors --------------------------------------------------
    base_body = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "测试"}],
        "resolution": "480P",
        "duration": 4,
        "ratio": "16:9",
    }
    status, _, payload = request("POST", "/v1/videos", body={**base_body, "duration": 3})
    record("duration=3 → 400", status == 400, f"HTTP {status} {payload[:120]}")
    status, _, payload = request("POST", "/v1/videos", body={**base_body, "duration": 16})
    record("duration=16 → 400", status == 400, f"HTTP {status} {payload[:120]}")
    status, _, payload = request("POST", "/v1/videos", body={**base_body, "resolution": "2K"})
    record("resolution=2K → 400", status == 400, f"HTTP {status} {payload[:120]}")
    status, _, payload = request("POST", "/v1/videos", body={**base_body, "ratio": "adaptive"})
    record("t2va ratio=adaptive → 400", status == 400, f"HTTP {status} {payload[:120]}")

    # -- SSRF guard ----------------------------------------------------------
    status, _, payload = request("POST", "/v1/videos/sync", body={"model": "MiniMax-H3", "content": [
        {"type": "text", "text": "测试"},
        {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8091/health"}, "role": "last_frame"},
    ], "resolution": "480P", "duration": 4, "ratio": "adaptive"})
    blocked = status == 400 and (b"private" in payload or b"loopback" in payload or b"reserved" in payload)
    record("SSRF: last_frame 指向 127.0.0.1 → 400", blocked, f"HTTP {status} {payload[:160]}")

    # -- async t2va full cycle (also the persistence fixture) ---------------
    status, _, payload = request("POST", "/v1/videos", body=base_body)
    job = json.loads(payload) if status == 202 else {}
    ok = status == 202 and job.get("id", "").startswith("h3_") and job.get("status") == "queued"
    record("POST /v1/videos 创建 t2va 任务", ok, f"HTTP {status} id={job.get('id')}")
    t2va_id = job.get("id")
    job = wait_job(t2va_id) if t2va_id else {"status": "skipped"}
    record("轮询 t2va 任务至完成", job.get("status") == "completed", f"最终状态 {job.get('status')} {str(job.get('error'))[:200]}")
    if t2va_id and job.get("status") == "completed":
        status, headers, payload = request("GET", f"/v1/videos/{t2va_id}/content")
        record("下载 t2va 内容", status == 200 and is_mp4(payload), f"HTTP {status}, {len(payload)} bytes, CT={headers.get('Content-Type')}")
        status, _, payload = request("GET", "/v1/videos")
        ids = [item["id"] for item in json.loads(payload).get("data", [])]
        record("任务出现在列表", t2va_id in ids, f"HTTP {status}, 列表 {len(ids)} 条")
    else:
        record("下载 t2va 内容", False, "前置任务未完成，跳过")
        record("任务出现在列表", False, "前置任务未完成，跳过")

    # -- sync generations ----------------------------------------------------
    def sync_case(name, content):
        body = {"model": "MiniMax-H3", "content": content, "resolution": "480P", "duration": 4, "ratio": "adaptive",
                "generation_config": {"num_inference_steps": 2, "seed": 42}}
        status, headers, payload = request("POST", "/v1/videos/sync", body=body)
        ok = status == 200 and is_mp4(payload)
        record(name, ok, f"HTTP {status}, {len(payload)} bytes, CT={headers.get('Content-Type')}")
        return ok

    sync_case("sync 首帧图生视频", [
        {"type": "text", "text": "画面从首帧自然展开，镜头缓慢推进。"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "first-frame.png", "image/png")}, "role": "first_frame"},
    ])
    sync_case("sync 尾帧图生视频", [
        {"type": "text", "text": "镜头自然推进，画面落在结尾帧上。"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "last-frame.png", "image/png")}, "role": "last_frame"},
    ])
    sync_case("sync 首尾帧图生视频", [
        {"type": "text", "text": "画面从首帧平滑过渡到尾帧。"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "first-frame.png", "image/png")}, "role": "first_frame"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "last-frame.png", "image/png")}, "role": "last_frame"},
    ])
    sync_case("sync 图片+音频参考生成", [
        {"type": "text", "text": "主体跟随参考音频自然运动。"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "fl2va-input.png", "image/png")}, "role": "reference_image"},
        {"type": "audio_url", "audio_url": {"url": data_url(OUT / "ref-audio.mp3", "audio/mpeg")}, "role": "reference_audio"},
    ])
    sync_case("sync 多图参考生成（2 张参考图）", [
        {"type": "text", "text": "两个参考主体同框自然互动。"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "first-frame.png", "image/png")}, "role": "reference_image"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "last-frame.png", "image/png")}, "role": "reference_image"},
    ])

    # -- create + delete ------------------------------------------------------
    status, _, payload = request("POST", "/v1/videos", body={"model": "MiniMax-H3", "content": [
        {"type": "text", "text": "测试删除流程。"},
        {"type": "image_url", "image_url": {"url": data_url(OUT / "last-frame.png", "image/png")}, "role": "last_frame"},
    ], "resolution": "480P", "duration": 4, "ratio": "adaptive", "generation_config": {"num_inference_steps": 2}})
    del_job = json.loads(payload) if status == 202 else {}
    record("POST /v1/videos 创建待删除任务", status == 202 and bool(del_job.get("id")), f"HTTP {status} id={del_job.get('id')}")
    if del_job.get("id"):
        status, _, _ = request("DELETE", "/v1/videos/" + del_job["id"])
        record("DELETE 任务 → 204", status == 204, f"HTTP {status}")
        status, _, _ = request("GET", "/v1/videos/" + del_job["id"])
        record("删除后 GET → 404", status == 404, f"HTTP {status}")

    # -- unknown ids -----------------------------------------------------------
    status, _, _ = request("GET", "/v1/videos/h3_does_not_exist")
    record("GET 未知任务 → 404", status == 404, f"HTTP {status}")
    status, _, _ = request("GET", "/v1/videos/h3_does_not_exist/content")
    record("GET 未知任务内容 → 404", status == 404, f"HTTP {status}")
    status, _, _ = request("DELETE", "/v1/videos/h3_does_not_exist")
    record("DELETE 未知任务 → 404", status == 404, f"HTTP {status}")

    # -- persistence across gateway restart ------------------------------------
    if t2va_id:
        subprocess.run(["docker", "restart", "minimax-h3-gateway"], check=True, shell=False)
        deadline = time.time() + 180
        up = False
        while time.time() < deadline:
            try:
                status, _, _ = request("GET", "/health", key="", timeout=5)
                if status == 200:
                    up = True
                    break
            except Exception:
                pass
            time.sleep(5)
        record("网关重启后恢复", up, f"health={'200' if up else 'timeout'}")
        status, _, payload = request("GET", "/v1/videos/" + t2va_id)
        ok = status == 200 and json.loads(payload).get("status") == "completed"
        record("重启后历史任务仍在", ok, f"HTTP {status}")
        status, _, payload = request("GET", f"/v1/videos/{t2va_id}/content")
        record("重启后视频文件仍可下载", status == 200 and is_mp4(payload), f"HTTP {status}, {len(payload)} bytes")
    else:
        record("持久化验证", False, "无已完成任务可验证，跳过")

    # -- report -----------------------------------------------------------------
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    lines = ["# MiniMax H3 统一接口测试报告", "",
             f"- 测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 目标：http://{HOST}:{PORT}",
             f"- 结果：{passed}/{len(RESULTS)} 通过", "",
             "| # | 用例 | 结果 | 详情 |", "|---|------|------|------|"]
    for index, (name, ok, detail) in enumerate(RESULTS, 1):
        lines.append(f"| {index} | {name} | {'✅ 通过' if ok else '❌ 失败'} | {detail.replace('|', '/')} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{passed}/{len(RESULTS)} passed — report: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
