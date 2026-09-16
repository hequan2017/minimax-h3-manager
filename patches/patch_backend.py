#!/usr/bin/env python3
"""Patch applier: enables first/last keyframe FL2VA and image-only Ref2VA
inside a running vllm-omni MiniMax H3 container, then the result can be
committed as the deployment image.

Usage (on the GPU host, against a running backend container):
    python3 patch_backend.py            # phase 1: keyframe FL2VA + serving layer
    python3 patch_backend.py phase2     # follow-up: decode keyframe uploads by content
    python3 patch_backend.py phase3     # image-only Ref2VA (audio reference optional)

Targets (override via env): H3_PATCH_CONTAINER (default minimax-h3-fl2va),
H3_PATCH_WORK (default /tmp/h3-patch-work).

Every edit is anchored: if an anchor is missing or duplicated the script
aborts BEFORE installing anything. Patched files are syntax-checked inside
the container before being moved into place. Originals are backed up under
$H3_PATCH_WORK/orig (host, authoritative) and /tmp/h3_patch_backup_serving_video.py
/ /tmp/h3_patch_backup_pipeline_minimax_h3.py (container).

After patching, persist with:
    docker commit <container> vllm/vllm-omni:minimax-h3
    docker compose up -d
"""

import os
import pathlib
import shutil
import subprocess
import sys

HOST = os.environ.get("H3_PATCH_CONTAINER", "minimax-h3-fl2va")
WORK = pathlib.Path(os.environ.get("H3_PATCH_WORK", "/tmp/h3-patch-work"))
SRC = WORK / "src"
PATCHED = WORK / "patched"
ORIG = WORK / "orig"
MARKER = WORK / ".container_backup_done"
PKG = "/usr/local/lib/python3.12/dist-packages/vllm_omni"

SERVING_PATH = PKG + "/entrypoints/openai/serving_video.py"
PIPELINE_PATH = PKG + "/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
FILES = {
    "serving_video.py": SERVING_PATH,
    "pipeline_minimax_h3.py": PIPELINE_PATH,
}


def apply_edits(name: str, text: str, edits):
    for index, (old, new) in enumerate(edits):
        count = text.count(old)
        if count != 1:
            print(f"[{name}] edit #{index} anchor matched {count} times (expected 1):")
            print("---- anchor ----")
            print(old[:400])
            sys.exit(f"aborting {name}: anchor #{index} not unique")
        text = text.replace(old, new, 1)
        print(f"[{name}] edit #{index} applied")
    return text


def install_file(name: str, local_text: str, cpath: str) -> None:
    """Stage one patched file into the container: copy out, patch, check, install."""
    staged = pathlib.Path("/tmp") / (name + ".phase")
    staged.write_text(local_text)
    subprocess.run(
        ["docker", "cp", str(staged), HOST + ":/tmp/" + name + ".patched"],
        check=True,
        shell=False,
    )
    subprocess.run(
        ["docker", "exec", HOST, "python3", "-m", "py_compile", "/tmp/" + name + ".patched"],
        check=True,
        shell=False,
    )
    subprocess.run(
        ["docker", "exec", HOST, "cp", "/tmp/" + name + ".patched", cpath],
        check=True,
        shell=False,
    )
    print(f"[{name}] installed -> {cpath}")


def main() -> None:
    for directory in (SRC, PATCHED, ORIG):
        directory.mkdir(parents=True, exist_ok=True)

    sources = {}
    for name, cpath in FILES.items():
        subprocess.run(
            ["docker", "cp", HOST + ":" + cpath, str(SRC / name)],
            check=True,
            shell=False,
        )
        shutil.copy2(SRC / name, ORIG / name)
        sources[name] = (SRC / name).read_text()

    # ---------------- serving_video.py ----------------
    serving_edits = [
        (
            """        input_image = None if reference_image is None else reference_image.data
        input_video = None if reference_video is None else reference_video.data
        if input_image is not None and input_video is not None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Provide either an image reference or a video reference, not both.",
            )
""",
            """        input_image = None if reference_image is None else reference_image.data
        input_video = None if reference_video is None else reference_video.data
        request_task = None
        if isinstance(request.extra_params, dict):
            request_task = str(request.extra_params.get("task") or "").lower()
        if input_image is not None and input_video is not None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Provide either an image reference or a video reference, not both.",
            )
        if (
            request_task == "fl2va"
            and isinstance(input_video, (list, tuple))
            and input_video
            and all(str(item).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")) for item in input_video)
        ):
            # fl2va input_references uploads are first/last keyframe images,
            # not reference videos; hand them to the pipeline as image list.
            keyframe_images = [Image.open(str(item)).convert("RGB") for item in input_video]
            if len(keyframe_images) > 2:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="fl2va accepts at most two keyframe images (first_frame and last_frame).",
                )
            input_image = keyframe_images if len(keyframe_images) > 1 else keyframe_images[0]
            input_video = None
""",
        ),
        (
            """        if input_image is not None and vp.width is not None and vp.height is not None:
            target_size = (vp.width, vp.height)
            if input_image.size != target_size:
                input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
""",
            """        if (
            input_image is not None
            and not isinstance(input_image, list)
            and vp.width is not None
            and vp.height is not None
        ):
            target_size = (vp.width, vp.height)
            if input_image.size != target_size:
                input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
""",
        ),
    ]

    # ---------------- pipeline_minimax_h3.py ----------------
    pipeline_edits = [
        (
            """    raise TypeError(f"unsupported MiniMax H3 image input {type(value)!r}")
""",
            """    raise TypeError(f"unsupported MiniMax H3 image input {type(value)!r}")


def _load_images(value: Any) -> list[Image.Image]:
    if isinstance(value, (list, tuple)):
        return [_load_image(item) for item in value]
    return [_load_image(value)]


def _resolve_keyframe_frame_indices(extra: dict, image_count: int) -> list[int]:
    raw = extra.get("frame_indices", extra.get("keyframe_frame_indices"))
    if raw is None:
        return [0]
    indices = [int(item) for item in raw] if isinstance(raw, (list, tuple)) else [int(raw)]
    if not indices:
        return [0]
    if tuple(indices) not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            f"fl2va frame_indices must be one of {MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got {indices!r}"
        )
    if len(indices) != image_count:
        raise ValueError(f"fl2va frame_indices {indices!r} must have one entry per condition image ({image_count})")
    return indices
""",
        ),
        (
            """        raw_image = multi_modal_data.get("image")
        raw_videos = multi_modal_data.get("video")
        image = _load_image(raw_image) if raw_image is not None else None
        if task == "fl2va" and image is None:
            raise ValueError(f"{task} requires multi_modal_data.image")
""",
            """        raw_image = multi_modal_data.get("image")
        raw_videos = multi_modal_data.get("video")
        images = _load_images(raw_image) if raw_image is not None else []
        if task == "fl2va" and not images:
            raise ValueError(f"{task} requires multi_modal_data.image")
        keyframe_frame_indices = (
            _resolve_keyframe_frame_indices(extra, len(images)) if task == "fl2va" and images else None
        )
        image = images[0] if images else None
""",
        ),
        (
            """        prepared_image = image
        if task == "fl2va" and image is not None:
            prepared_image = image.resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )
""",
            """        prepared_image = image
        prepared_images: list = []
        if task == "fl2va" and image is not None:
            prepared_images = [item.resize((width, height), Image.Resampling.LANCZOS) for item in images]
""",
        ),
        (
            """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_image,
                prepared_videos=prepared_videos,
            )
""",
            """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_images if task == "fl2va" else prepared_image,
                prepared_videos=prepared_videos,
            )
""",
        ),
        (
            """            elif prepared_image is not None:
                visual_condition = self._encode_visual_condition(prepared_image)
                visual_shape = (
                    1,
                    prepared_image.height // 16,
                    prepared_image.width // 16,
                )
""",
            """            elif prepared_images:
                condition_rows = []
                condition_shapes = []
                for item in prepared_images:
                    condition_rows.append(self._encode_visual_condition(item))
                    condition_shapes.append((1, item.height // 16, item.width // 16))
                visual_condition = torch.cat(condition_rows, dim=0) if len(condition_rows) > 1 else condition_rows[0]
                if len(condition_shapes) == 1:
                    visual_shape = condition_shapes[0]
                else:
                    visual_shapes = condition_shapes
            elif prepared_image is not None:
                visual_condition = self._encode_visual_condition(prepared_image)
                visual_shape = (
                    1,
                    prepared_image.height // 16,
                    prepared_image.width // 16,
                )
""",
        ),
        (
            """        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
            """        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
        keyframe_frame_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
        ),
        (
            """                keyframe_frame_indices=[0] if task == "fl2va" else None,
""",
            """                keyframe_frame_indices=keyframe_frame_indices if task == "fl2va" else None,
""",
        ),
        (
            """            visual_condition_shapes=visual_shapes,
            audio_condition_lengths=audio_lengths,
        )
""",
            """            visual_condition_shapes=visual_shapes,
            audio_condition_lengths=audio_lengths,
            keyframe_frame_indices=keyframe_frame_indices,
        )
""",
        ),
        (
            """                if image is None:
                    raise ValueError(f"{task} requires one image")
                vision = self.processor.image_processor(
                    images=[image],
                    return_tensors="pt",
                )
                image_grid = vision["image_grid_thw"]
                merge = int(self.processor.image_processor.merge_size) ** 2
                image_tokens = int(image_grid[0].prod().item()) // merge
                if task == "fl2va":
                    ids = minimax_h3_multi_image_presentation_ids(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=[image_tokens],
                    )
                    tags = minimax_h3_multi_image_presentation_token_tags(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=[image_tokens],
                    )
                else:
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=[("image", 1), ("audio", 1)],
                        image_token_count=image_tokens,
                    )
""",
            """                prompt_images = image if isinstance(image, list) else [image]
                if not prompt_images or prompt_images[0] is None:
                    raise ValueError(f"{task} requires one image")
                vision = self.processor.image_processor(
                    images=prompt_images,
                    return_tensors="pt",
                )
                image_grid = vision["image_grid_thw"]
                merge = int(self.processor.image_processor.merge_size) ** 2
                image_token_counts = [
                    int(image_grid[index].prod().item()) // merge for index in range(len(prompt_images))
                ]
                if task == "fl2va":
                    ids = minimax_h3_multi_image_presentation_ids(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=image_token_counts,
                    )
                    tags = minimax_h3_multi_image_presentation_token_tags(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=image_token_counts,
                    )
                else:
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=[("image", 1), ("audio", 1)],
                        image_token_count=image_token_counts[0],
                    )
""",
        ),
    ]

    patched = {}
    for name, edits in (
        ("serving_video.py", serving_edits),
        ("pipeline_minimax_h3.py", pipeline_edits),
    ):
        text = sources[name]
        needs_constant_import = name == "pipeline_minimax_h3.py" and "MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES" not in text
        if needs_constant_import:
            if "from .packed_sequence import (" in text:
                text = text.replace(
                    "from .packed_sequence import (",
                    "from .packed_sequence import (\n    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,",
                    1,
                )
                print(f"[{name}] added constant to existing packed_sequence import")
            else:
                anchor = "from .packed_sequence import"
                if anchor not in text:
                    sys.exit(f"aborting {name}: no .packed_sequence import found for constant")
                text = text.replace(
                    anchor,
                    "from .packed_sequence import MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES\n" + anchor,
                    1,
                )
                print(f"[{name}] added standalone packed_sequence constant import")
        patched[name] = apply_edits(name, text, edits)

    # write patched files and syntax-check inside the container
    for name in FILES:
        (PATCHED / name).write_text(patched[name])
        subprocess.run(
            ["docker", "cp", str(PATCHED / name), HOST + ":/tmp/" + name + ".patched"],
            check=True,
            shell=False,
        )
        subprocess.run(
            ["docker", "exec", HOST, "python3", "-m", "py_compile", "/tmp/" + name + ".patched"],
            check=True,
            shell=False,
        )
        print(f"[{name}] syntax check OK")

    # install: one-time backup of the pristine files inside the container,
    # then move the patched files into place
    if not MARKER.exists():
        subprocess.run(
            ["docker", "exec", HOST, "cp", SERVING_PATH, "/tmp/h3_patch_backup_serving_video.py"],
            check=True,
            shell=False,
        )
        subprocess.run(
            ["docker", "exec", HOST, "cp", PIPELINE_PATH, "/tmp/h3_patch_backup_pipeline_minimax_h3.py"],
            check=True,
            shell=False,
        )
        MARKER.write_text("done\n")
        print("container-side originals backed up")
    subprocess.run(
        ["docker", "exec", HOST, "cp", "/tmp/serving_video.py.patched", SERVING_PATH],
        check=True,
        shell=False,
    )
    print("[serving_video.py] installed")
    subprocess.run(
        ["docker", "exec", HOST, "cp", "/tmp/pipeline_minimax_h3.py.patched", PIPELINE_PATH],
        check=True,
        shell=False,
    )
    print("[pipeline_minimax_h3.py] installed")
    print("ALL PATCHES INSTALLED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "phase4":
        # Multi-image Ref2VA: the packed builder and Qwen presentation already
        # accept N image blocks with per-image shapes/labels; only the glue
        # layer collapsed reference images to one. Prepare every reference
        # image, emit one <Picture i> slot per image, and build one image
        # ref-block per image (audio block appended only when present).
        name = "pipeline_minimax_h3.py"
        cpath = FILES[name]
        staged = pathlib.Path("/tmp") / (name + ".phase4out")
        subprocess.run(
            ["docker", "cp", HOST + ":" + cpath, str(staged)],
            check=True,
            shell=False,
        )
        text = staged.read_text()
        edits = [
            (
                """        prepared_image = image
        prepared_images: list = []
        if task == "fl2va" and image is not None:
            prepared_images = [item.resize((width, height), Image.Resampling.LANCZOS) for item in images]
        elif task == "ref2va" and image is not None:
            ref_width, ref_height = _reference_image_shape(image)
            prepared_image = image.resize(
                (ref_width, ref_height),
                Image.Resampling.LANCZOS,
            )
""",
                """        prepared_image = None
        prepared_images: list = []
        if task == "fl2va" and image is not None:
            prepared_images = [item.resize((width, height), Image.Resampling.LANCZOS) for item in images]
        elif task == "ref2va" and image is not None:
            for item in images:
                ref_width, ref_height = _reference_image_shape(item)
                prepared_images.append(item.resize((ref_width, ref_height), Image.Resampling.LANCZOS))
""",
            ),
            (
                """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_images if task == "fl2va" else prepared_image,
                prepared_videos=prepared_videos,
                has_ref_audio=task == "ref2va" and multi_modal_data.get("audio") is not None,
            )
""",
                """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_images if task in ("fl2va", "ref2va") else prepared_image,
                prepared_videos=prepared_videos,
                has_ref_audio=task == "ref2va" and multi_modal_data.get("audio") is not None,
            )
""",
            ),
            (
                """                else:
                    condition_labels = [("image", 1)] + ([("audio", 1)] if has_ref_audio else [])
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts[0],
                    )
""",
                """                else:
                    condition_labels = [("image", index) for index in range(1, len(prompt_images) + 1)]
                    if has_ref_audio:
                        condition_labels.append(("audio", 1))
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts,
                    )
""",
            ),
            (
                """        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [{"kind": "image", "latent_h": ref_h, "latent_w": ref_w}]
                if ref_audio_t is not None:
                    ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
""",
                """        if task == "ref2va":
            if ref_blocks is None:
                shapes = visual_condition_shapes
                if shapes is None and visual_condition_shape is not None:
                    shapes = [visual_condition_shape]
                if not shapes:
                    raise ValueError("ref2va condition metadata is missing")
                ref_blocks = [
                    {"kind": "image", "latent_h": int(shape[1]), "latent_w": int(shape[2])}
                    for shape in shapes
                ]
                if ref_audio_t is not None:
                    ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
""",
            ),
        ]
        patched_text = apply_edits(name, text, edits)

        # serving layer: partition input_references into images vs real videos
        sname = "serving_video.py"
        scpath = FILES[sname]
        sstaged = pathlib.Path("/tmp") / (sname + ".phase4out")
        subprocess.run(
            ["docker", "cp", HOST + ":" + scpath, str(sstaged)],
            check=True,
            shell=False,
        )
        stext = sstaged.read_text()
        sold = """        if request_task == "fl2va" and isinstance(input_video, (list, tuple)) and input_video:
            # fl2va input_references uploads are first/last keyframe images
            # (persisted with a .mp4 suffix regardless of content), not
            # reference videos; decode and hand them to the pipeline as an
            # image list.
            keyframe_images = []
            for item in input_video:
                try:
                    keyframe_images.append(Image.open(str(item)).convert("RGB"))
                except Exception:
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST.value,
                        detail="fl2va input_references must be image files (first_frame/last_frame).",
                    )
            if len(keyframe_images) > 2:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="fl2va accepts at most two keyframe images (first_frame and last_frame).",
                )
            input_image = keyframe_images if len(keyframe_images) > 1 else keyframe_images[0]
            input_video = None
"""
        snew = """        if request_task in ("fl2va", "ref2va") and isinstance(input_video, (list, tuple)) and input_video:
            # keyframe/reference uploads are images persisted with a .mp4
            # suffix; decode them by content and hand them to the pipeline as
            # an image list. True video files keep the reference-video path.
            keyframe_images = []
            video_paths = []
            for item in input_video:
                try:
                    keyframe_images.append(Image.open(str(item)).convert("RGB"))
                except Exception:
                    video_paths.append(item)
            if keyframe_images and video_paths:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="input_references cannot mix image and video files.",
                )
            if keyframe_images:
                if request_task == "fl2va" and len(keyframe_images) > 2:
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST.value,
                        detail="fl2va accepts at most two keyframe images (first_frame and last_frame).",
                    )
                if request_task == "ref2va" and len(keyframe_images) > 4:
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST.value,
                        detail="ref2va accepts at most four reference images on this deployment.",
                    )
                input_image = keyframe_images if len(keyframe_images) > 1 else keyframe_images[0]
                input_video = None
            elif request_task == "fl2va":
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="fl2va input_references must be image files (first_frame/last_frame).",
                )
"""
        spatched = apply_edits(sname, stext, [(sold, snew)])
        install_file(sname, spatched, scpath)
        print("PHASE4 COMPLETE")
    elif len(sys.argv) > 1 and sys.argv[1] == "phase4fix":
        # Recovery for the phase4 variable-reuse bug (serving content was
        # installed over the pipeline file): rebuild the pipeline from the
        # intact copy in another container, then re-apply phase4 edits.
        # H3_PATCH_CONTAINER = broken target (e.g. minimax-h3-ref2va)
        # H3_PATCH_SRC_CONTAINER = intact source (default minimax-h3-fl2va)
        src_host = os.environ.get("H3_PATCH_SRC_CONTAINER", "minimax-h3-fl2va")
        name = "pipeline_minimax_h3.py"
        cpath = FILES[name]
        staged = pathlib.Path("/tmp") / (name + ".phase4fixout")
        subprocess.run(
            ["docker", "cp", src_host + ":" + cpath, str(staged)],
            check=True,
            shell=False,
        )
        text = staged.read_text()
        edits = [
            (
                """        prepared_image = image
        prepared_images: list = []
        if task == "fl2va" and image is not None:
            prepared_images = [item.resize((width, height), Image.Resampling.LANCZOS) for item in images]
        elif task == "ref2va" and image is not None:
            ref_width, ref_height = _reference_image_shape(image)
            prepared_image = image.resize(
                (ref_width, ref_height),
                Image.Resampling.LANCZOS,
            )
""",
                """        prepared_image = None
        prepared_images: list = []
        if task == "fl2va" and image is not None:
            prepared_images = [item.resize((width, height), Image.Resampling.LANCZOS) for item in images]
        elif task == "ref2va" and image is not None:
            for item in images:
                ref_width, ref_height = _reference_image_shape(item)
                prepared_images.append(item.resize((ref_width, ref_height), Image.Resampling.LANCZOS))
""",
            ),
            (
                """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_images if task == "fl2va" else prepared_image,
                prepared_videos=prepared_videos,
                has_ref_audio=task == "ref2va" and multi_modal_data.get("audio") is not None,
            )
""",
                """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_images if task in ("fl2va", "ref2va") else prepared_image,
                prepared_videos=prepared_videos,
                has_ref_audio=task == "ref2va" and multi_modal_data.get("audio") is not None,
            )
""",
            ),
            (
                """                else:
                    condition_labels = [("image", 1)] + ([("audio", 1)] if has_ref_audio else [])
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts[0],
                    )
""",
                """                else:
                    condition_labels = [("image", index) for index in range(1, len(prompt_images) + 1)]
                    if has_ref_audio:
                        condition_labels.append(("audio", 1))
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts,
                    )
""",
            ),
            (
                """        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [{"kind": "image", "latent_h": ref_h, "latent_w": ref_w}]
                if ref_audio_t is not None:
                    ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
""",
                """        if task == "ref2va":
            if ref_blocks is None:
                shapes = visual_condition_shapes
                if shapes is None and visual_condition_shape is not None:
                    shapes = [visual_condition_shape]
                if not shapes:
                    raise ValueError("ref2va condition metadata is missing")
                ref_blocks = [
                    {"kind": "image", "latent_h": int(shape[1]), "latent_w": int(shape[2])}
                    for shape in shapes
                ]
                if ref_audio_t is not None:
                    ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
""",
            ),
        ]
        patched_text = apply_edits(name, text, edits)
        install_file(name, patched_text, cpath)
        print("PHASE4FIX COMPLETE")
    elif len(sys.argv) > 1 and sys.argv[1] == "phase3":
        # Image-only Ref2VA: the official input matrix allows reference images
        # without audio (audio-only is what gets rejected). Older vllm-omni
        # builds hard-require audio for image Ref2VA; relax it while keeping
        # image+audio behavior byte-identical when audio is present.
        name = "pipeline_minimax_h3.py"
        cpath = FILES[name]
        staged = pathlib.Path("/tmp") / (name + ".phase3out")
        subprocess.run(
            ["docker", "cp", HOST + ":" + cpath, str(staged)],
            check=True,
            shell=False,
        )
        text = staged.read_text()
        edits = [
            (
                """            if task == "ref2va" and raw_videos is None:
                raw_audio = multi_modal_data.get("audio")
                if raw_audio is None:
                    raise ValueError("image Ref2VA requires multi_modal_data.audio")
                audio_condition, ref_audio_t = self._encode_audio_condition(_load_audio(raw_audio))
""",
                """            if task == "ref2va" and raw_videos is None:
                raw_audio = multi_modal_data.get("audio")
                if raw_audio is not None:
                    audio_condition, ref_audio_t = self._encode_audio_condition(_load_audio(raw_audio))
""",
            ),
            (
                """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_images if task == "fl2va" else prepared_image,
                prepared_videos=prepared_videos,
            )
""",
                """            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_images if task == "fl2va" else prepared_image,
                prepared_videos=prepared_videos,
                has_ref_audio=task == "ref2va" and multi_modal_data.get("audio") is not None,
            )
""",
            ),
            (
                """    def encode_prompt(
        self,
        *,
        task: str,
        prompt: str,
        image: Image.Image | None,
        prepared_videos: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
                """    def encode_prompt(
        self,
        *,
        task: str,
        prompt: str,
        image: Image.Image | None,
        prepared_videos: list[dict[str, Any]] | None = None,
        has_ref_audio: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
            ),
            (
                """                else:
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=[("image", 1), ("audio", 1)],
                        image_token_count=image_token_counts[0],
                    )
""",
                """                else:
                    condition_labels = [("image", 1)] + ([("audio", 1)] if has_ref_audio else [])
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts[0],
                    )
""",
            ),
            (
                """        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None or ref_audio_t is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [
                    {"kind": "image", "latent_h": ref_h, "latent_w": ref_w},
                    {"kind": "audio", "ref_audio_t": ref_audio_t},
                ]
""",
                """        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [{"kind": "image", "latent_h": ref_h, "latent_w": ref_w}]
                if ref_audio_t is not None:
                    ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
""",
            ),
        ]
        patched_text = apply_edits(name, text, edits)
        install_file(name, patched_text, cpath)
        print("PHASE3 COMPLETE")
    elif len(sys.argv) > 1 and sys.argv[1] == "phase2":
        # Follow-up fix: _persist_uploaded_video_references stores uploads with
        # a .mp4 suffix even for images, so decoding by suffix never matches.
        # Decode by content instead (PIL Image.open with a clean 400 on error).
        name = "serving_video.py"
        cpath = FILES[name]
        staged = pathlib.Path("/tmp") / (name + ".phase2out")
        subprocess.run(
            ["docker", "cp", HOST + ":" + cpath, str(staged)],
            check=True,
            shell=False,
        )
        text = staged.read_text()
        old = """        if (
            request_task == "fl2va"
            and isinstance(input_video, (list, tuple))
            and input_video
            and all(str(item).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")) for item in input_video)
        ):
            # fl2va input_references uploads are first/last keyframe images,
            # not reference videos; hand them to the pipeline as image list.
            keyframe_images = [Image.open(str(item)).convert("RGB") for item in input_video]
"""
        new = """        if request_task == "fl2va" and isinstance(input_video, (list, tuple)) and input_video:
            # fl2va input_references uploads are first/last keyframe images
            # (persisted with a .mp4 suffix regardless of content), not
            # reference videos; decode and hand them to the pipeline as an
            # image list.
            keyframe_images = []
            for item in input_video:
                try:
                    keyframe_images.append(Image.open(str(item)).convert("RGB"))
                except Exception:
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST.value,
                        detail="fl2va input_references must be image files (first_frame/last_frame).",
                    )
"""
        patched_text = apply_edits(name, text, [(old, new)])
        install_file(name, patched_text, cpath)
        print("PHASE2 COMPLETE")
    else:
        main()
