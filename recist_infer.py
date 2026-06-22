#!/usr/bin/env python3
"""Unified RECIST-prompted inference entry for RECISTto3D.

This script intentionally shares only I/O and RECIST geometry. Each model keeps
its own image preprocessing in its own runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PACK_ROOT = Path(__file__).resolve().parent
MEDSAM2_ROOT = PACK_ROOT / "MedSAM2"
NNINTERACTIVE_ROOT = PACK_ROOT / "nnInteractive"


@dataclass
class LoadedImage:
    array: np.ndarray
    spacing: np.ndarray
    sitk_image: object | None = None
    source_key: str | None = None


@dataclass
class InferenceResult:
    segs: np.ndarray
    boxes_xyzxyz: np.ndarray
    metadata: dict


@dataclass
class PromptSpec:
    label: int
    kind: str
    z: int
    box_xyxy: np.ndarray | None = None
    points_xy: np.ndarray | None = None
    negative_points_xy: np.ndarray | None = None
    z_min: int | None = None
    z_max: int | None = None


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def add_to_syspath(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def assert_sitk_geometry_matches(recist_image, reference_image, recist_path: Path, *, atol: float = 1e-5) -> None:
    mismatches = []
    if recist_image.GetSize() != reference_image.GetSize():
        mismatches.append(f"size recist={recist_image.GetSize()} image={reference_image.GetSize()}")
    if not np.allclose(recist_image.GetSpacing(), reference_image.GetSpacing(), atol=atol):
        mismatches.append(f"spacing recist={recist_image.GetSpacing()} image={reference_image.GetSpacing()}")
    if not np.allclose(recist_image.GetOrigin(), reference_image.GetOrigin(), atol=atol):
        mismatches.append(f"origin recist={recist_image.GetOrigin()} image={reference_image.GetOrigin()}")
    if not np.allclose(recist_image.GetDirection(), reference_image.GetDirection(), atol=atol):
        mismatches.append(f"direction recist={recist_image.GetDirection()} image={reference_image.GetDirection()}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "RECIST NIfTI geometry must match image NIfTI geometry. "
            f"{recist_path} mismatch: {details}. "
            "Fix/resample the image and RECIST mask to the same grid before inference, "
            "or use --recist-space index only when the mask header is wrong but the array is already image-index aligned."
        )


def load_npz_array(path: Path, preferred_keys: Iterable[str], explicit_key: str | None) -> tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=True)
    if explicit_key is not None:
        if explicit_key not in data:
            raise KeyError(f"{path} does not contain key '{explicit_key}'. Available keys: {list(data.keys())}")
        return np.asarray(data[explicit_key]), explicit_key

    for key in preferred_keys:
        if key in data:
            return np.asarray(data[key]), key

    keys = list(data.keys())
    if not keys:
        raise ValueError(f"{path} has no arrays")
    return np.asarray(data[keys[0]]), keys[0]


def as_3d_volume(array: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None]
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[0] == 1:
        return array[0]
    raise ValueError(f"{name} must be 2D, 3D, or 4D with first dim 1; got shape {array.shape}")


def load_image(path: Path, image_key: str | None, spacing_arg: str | None) -> LoadedImage:
    sitk_image = None
    source_key = None

    if path.suffix.lower() == ".npz":
        array, source_key = load_npz_array(path, ("imgs", "image", "arr_0"), image_key)
        data = np.load(path, allow_pickle=True)
        if "spacing" in data:
            spacing = np.asarray(data["spacing"], dtype=np.float32)
        else:
            spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    elif path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=True)
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    elif is_nifti(path):
        import SimpleITK as sitk

        sitk_image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(sitk_image)
        spacing_xyz = sitk_image.GetSpacing()
        spacing = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
    else:
        from PIL import Image

        with Image.open(path) as img:
            array = np.asarray(img.convert("L"))
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    if spacing_arg is not None:
        spacing = parse_spacing(spacing_arg)

    array = as_3d_volume(array, name="image")
    spacing = normalize_spacing(spacing)
    return LoadedImage(array=array, spacing=spacing, sitk_image=sitk_image, source_key=source_key)


def normalize_spacing(spacing: np.ndarray) -> np.ndarray:
    spacing = np.asarray(spacing, dtype=np.float32).reshape(-1)
    if spacing.size == 1:
        return np.array([spacing[0], spacing[0], spacing[0]], dtype=np.float32)
    if spacing.size != 3:
        raise ValueError(f"spacing must have 1 or 3 values, got {spacing}")
    if np.any(spacing <= 0):
        raise ValueError(f"spacing values must be positive, got {spacing}")
    return spacing


def parse_spacing(value: str) -> np.ndarray:
    parts = [float(x) for x in value.replace(",", " ").split()]
    return normalize_spacing(np.asarray(parts, dtype=np.float32))


def load_recist(
    image_path: Path,
    recist_path: Path | None,
    recist_key: str | None,
    recist_lines: list[str],
    shape: tuple[int, int, int],
    reference_sitk_image: object | None = None,
    recist_space: str = "strict",
) -> tuple[np.ndarray, str]:
    if recist_path is not None and recist_lines:
        raise ValueError("Use either --recist or --recist-line, not both.")

    if recist_path is not None:
        if recist_path.suffix.lower() == ".npz":
            recist, key = load_npz_array(recist_path, ("recist", "mask", "arr_0"), recist_key)
            source = f"{recist_path}:{key}"
        elif recist_path.suffix.lower() == ".npy":
            recist = np.load(recist_path, allow_pickle=True)
            source = str(recist_path)
        elif is_nifti(recist_path):
            import SimpleITK as sitk

            recist_image = sitk.ReadImage(str(recist_path))
            if recist_space == "strict" and reference_sitk_image is not None:
                assert_sitk_geometry_matches(recist_image, reference_sitk_image, recist_path)
            source = str(recist_path)
            recist = sitk.GetArrayFromImage(recist_image)
        else:
            from PIL import Image

            with Image.open(recist_path) as img:
                recist = np.asarray(img.convert("L"))
            source = str(recist_path)
        recist = as_3d_volume(recist, name="recist")
    elif recist_lines:
        parsed_lines = [parse_recist_line(line) for line in recist_lines]
        labels = [line[-1] for line in parsed_lines]
        duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
        if duplicate_labels:
            raise ValueError(
                "Each --recist-line must use a unique nonzero label for multi-lesion inference. "
                f"Duplicate labels: {duplicate_labels}. Use 'z,x1,y1,x2,y2,label'."
            )

        recist = np.zeros(shape, dtype=np.uint16)
        for z, x1, y1, x2, y2, label in parsed_lines:
            draw_recist_line(recist, z, x1, y1, x2, y2, label)
        source = "cli-recist-line"
    elif image_path.suffix.lower() == ".npz":
        recist, key = load_npz_array(image_path, ("recist",), recist_key)
        recist = as_3d_volume(recist, name="recist")
        source = f"{image_path}:{key}"
    else:
        raise ValueError("Provide --recist, --recist-line, or an input NPZ containing a 'recist' key.")

    if recist.shape != shape:
        raise ValueError(f"recist shape {recist.shape} must match image shape {shape}")

    if not np.issubdtype(recist.dtype, np.integer):
        recist = (recist > 0).astype(np.uint16)
    if not np.any(recist):
        hint = ""
        if recist_path is not None and is_nifti(recist_path) and recist_space == "strict":
            hint = (
                "; use --recist-space index only if the mask header is wrong but the array "
                "is already image-index aligned"
            )
        raise ValueError(f"RECIST mask is empty{hint}.")
    return recist.astype(np.uint16, copy=False), source


def parse_recist_line(value: str) -> tuple[int, int, int, int, int, int]:
    parts = [int(round(float(x))) for x in value.replace(",", " ").split()]
    if len(parts) == 5:
        z, x1, y1, x2, y2 = parts
        label = 1
    elif len(parts) == 6:
        z, x1, y1, x2, y2, label = parts
    else:
        raise argparse.ArgumentTypeError("--recist-line must be 'z,x1,y1,x2,y2' or 'z,x1,y1,x2,y2,label'")
    if label <= 0:
        raise argparse.ArgumentTypeError("--recist-line label must be a positive nonzero integer")
    return z, x1, y1, x2, y2, label


def draw_recist_line(mask: np.ndarray, z: int, x1: int, y1: int, x2: int, y2: int, label: int) -> None:
    d, h, w = mask.shape
    if not (0 <= z < d):
        raise ValueError(f"RECIST z={z} is outside image depth 0..{d - 1}")
    steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = np.rint(np.linspace(x1, x2, steps)).astype(int)
    ys = np.rint(np.linspace(y1, y2, steps)).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not np.any(valid):
        raise ValueError("RECIST line does not intersect the image")
    mask[z, ys[valid], xs[valid]] = label


def recist_coords_xy(recist_slice: np.ndarray) -> np.ndarray:
    ys, xs = np.where(recist_slice > 0)
    if len(xs) < 2:
        raise ValueError("RECIST line must contain at least two pixels")
    return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)


def recist_endpoints_xy(recist_slice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # EAY canonical prompt_utils.py uses the first and last rasterized RECIST
    # pixels, not a recomputed farthest pair, when converting RECIST to prompts.
    coords = recist_coords_xy(recist_slice)
    return coords[0], coords[-1]


def get_diameter(recist_slice: np.ndarray) -> float:
    p1, p2 = recist_endpoints_xy(recist_slice)
    return float(np.linalg.norm(p1 - p2))


def get_diameter_bbox(recist_slice: np.ndarray, shift: int = 0) -> np.ndarray:
    h, w = recist_slice.shape
    p1, p2 = recist_endpoints_xy(recist_slice)
    center = ((p1 + p2) / 2.0).astype(int)
    diameter = np.linalg.norm(p1 - p2)
    half_side = int(diameter / 2.0)

    x_min = max(0, int(center[0] - half_side) - shift)
    y_min = max(0, int(center[1] - half_side) - shift)
    x_max = min(w - 1, int(center[0] + half_side) + shift)
    y_max = min(h - 1, int(center[1] + half_side) + shift)

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def get_recist_5_points(recist_slice: np.ndarray, rng) -> np.ndarray:
    coords = recist_coords_xy(recist_slice)
    if len(coords) < 5:
        raise ValueError(f"Cannot sample 5 points; RECIST line only has {len(coords)} pixels.")
    idx = rng.choice(len(coords), size=5, replace=False)
    return coords[idx].astype(np.float32)


def filter_background_points(
    points_xy: np.ndarray,
    foreground_xy: np.ndarray,
    shape: tuple[int, int],
    *,
    forbidden_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Drop unsafe background points instead of clipping them onto foreground."""
    h, w = shape
    foreground_voxels = {
        (int(round(float(x))), int(round(float(y))))
        for x, y in np.asarray(foreground_xy, dtype=np.float32)
    }
    kept: list[list[float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in np.asarray(points_xy, dtype=np.float32):
        xf = float(x)
        yf = float(y)
        if xf < 0 or xf > w - 1 or yf < 0 or yf > h - 1:
            continue
        xi = int(round(xf))
        yi = int(round(yf))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        if (xi, yi) in foreground_voxels or (xi, yi) in seen:
            continue
        if forbidden_mask is not None and bool(forbidden_mask[yi, xi]):
            continue
        kept.append([xf, yf])
        seen.add((xi, yi))
    return np.asarray(kept, dtype=np.float32).reshape(-1, 2)


def get_recist_negative_points(recist_slice: np.ndarray, count: int) -> np.ndarray:
    """Adapt benchmark get_negative_pts geometry to RECIST endpoints only."""
    if count not in (4, 6):
        raise ValueError(f"RECIST negative point count must be 4 or 6, got {count}")

    p1, p2 = recist_endpoints_xy(recist_slice)
    axis = p2 - p1
    diameter = float(np.linalg.norm(axis))
    if diameter <= 0:
        raise ValueError("RECIST line must have a nonzero diameter")

    normal = np.array([-axis[1], axis[0]], dtype=np.float32) / diameter
    half_len = diameter / 2.0
    center = (p1 + p2) / 2.0

    candidate_points = np.stack(
        [
            p1 + normal * half_len,
            p1 - normal * half_len,
            p2 + normal * half_len,
            p2 - normal * half_len,
            center + normal * half_len * 1.2,
            center - normal * half_len * 1.2,
        ],
        axis=0,
    ).astype(np.float32)

    return filter_background_points(
        candidate_points[:count],
        recist_coords_xy(recist_slice),
        recist_slice.shape,
        forbidden_mask=recist_slice > 0,
    )


def prompt_specs_from_recist(
    recist: np.ndarray,
    spacing: np.ndarray,
    args,
    rng,
    *,
    target: str,
) -> list[PromptSpec]:
    specs = []
    labels = [int(x) for x in np.unique(recist) if x != 0]
    for label in labels:
        recist_per_label = (recist == label).astype(np.uint8)
        z_indices = np.unique(np.where(recist_per_label > 0)[0])
        if len(z_indices) != 1:
            raise ValueError(f"Label {label} must have RECIST pixels on exactly one z slice; got {z_indices.tolist()}")

        z_mid_orig = int(z_indices[0])
        recist_slice = recist_per_label[z_mid_orig]
        diameter = get_diameter(recist_slice)
        spacing_z = float(spacing[0])
        spacing_xy = float((spacing[1] + spacing[2]) / 2.0)
        multiplier = spacing_xy / spacing_z
        diameter_z = diameter / multiplier
        z_min = max(0, z_mid_orig - int(diameter_z / 2.0))
        z_max = min(recist.shape[0] - 1, z_mid_orig + int(diameter_z / 2.0))

        if target == "box":
            box_2d = get_diameter_bbox(recist_slice, shift=args.shift)
            specs.append(PromptSpec(label=label, kind="box", z=z_mid_orig, box_xyxy=box_2d, z_min=z_min, z_max=z_max))
        elif target in {"5_points", "5pos_4neg", "5pos_6neg"}:
            points = get_recist_5_points(recist_slice, rng)
            negative_points = None
            if target == "5pos_4neg":
                negative_points = get_recist_negative_points(recist_slice, count=4)
            elif target == "5pos_6neg":
                negative_points = get_recist_negative_points(recist_slice, count=6)
            specs.append(
                PromptSpec(
                    label=label,
                    kind="points",
                    z=z_mid_orig,
                    points_xy=points,
                    negative_points_xy=negative_points,
                    z_min=z_min,
                    z_max=z_max,
                )
            )
        else:
            raise ValueError(f"Unknown RECIST prompt target: {target}")
    return specs


def medsam2_z_range_for_prompt(prompt: PromptSpec) -> tuple[int, int]:
    if prompt.z_min is None or prompt.z_max is None:
        return int(prompt.z), int(prompt.z)
    return int(prompt.z_min), int(prompt.z_max)


def ensure_medsam2_uint8(image: np.ndarray, mode: str, window: str | None) -> np.ndarray:
    image = np.asarray(image)
    if mode == "preserve":
        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                "MedSAM2 expects images in [0, 255]. Use --intensity minmax or "
                "--intensity window --window MIN,MAX for raw CT inputs."
            )
        return np.clip(image, 0, 255).astype(np.uint8)

    image_f = image.astype(np.float32)
    if mode == "minmax":
        lo = float(np.nanmin(image_f))
        hi = float(np.nanmax(image_f))
    elif mode == "window":
        if window is None:
            raise ValueError("--intensity window requires --window MIN,MAX")
        vals = [float(x) for x in window.replace(",", " ").split()]
        if len(vals) != 2:
            raise ValueError("--window must have two values: MIN,MAX")
        lo, hi = vals
    else:
        raise ValueError(f"Unknown intensity mode: {mode}")

    if hi <= lo:
        raise ValueError(f"Invalid intensity range [{lo}, {hi}]")
    image_f = np.clip(image_f, lo, hi)
    image_f = (image_f - lo) / (hi - lo) * 255.0
    return np.rint(image_f).astype(np.uint8)


def resize_grayscale_to_rgb_and_resize(volume: np.ndarray, image_size: int) -> np.ndarray:
    from PIL import Image

    d, _, _ = volume.shape
    resized = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(volume[i].astype(np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size))
        resized[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
    return resized


def medsam2_preprocess(volume_uint8: np.ndarray, image_size: int = 512):
    import torch

    d, h, w = volume_uint8.shape
    if h != image_size or w != image_size:
        frames = resize_grayscale_to_rgb_and_resize(volume_uint8, image_size)
    else:
        frames = volume_uint8[:, None].repeat(3, axis=1).astype(np.float32)

    frames = frames / 255.0
    tensor = torch.from_numpy(frames)
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    tensor -= img_mean
    tensor /= img_std
    return tensor, h, w


def load_medsam2_predictor(args):
    add_to_syspath(MEDSAM2_ROOT)
    from huggingface_hub import hf_hub_download

    if args.model == "medsam2":
        from sam2.build_sam import build_sam2_video_predictor_npz

        filename = "medsam2_FLARE25_RECIST_baseline.pt"
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            local_ckpt_path = MEDSAM2_ROOT / "checkpoints" / filename
            ckpt_path = (
                local_ckpt_path
                if local_ckpt_path.exists()
                else hf_hub_download(
                    repo_id="wanglab/MedSAM2",
                    filename=filename,
                    cache_dir=str(MEDSAM2_ROOT / "checkpoints"),
                )
            )
        cfg_path = "//" + str(MEDSAM2_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml")
        device = args.device or "cuda"
        predictor = build_sam2_video_predictor_npz(cfg_path, ckpt_path, device=device)
        name = "MedSAM2"
    else:
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor_npz

        filename = "eff_medsam2_small_FLARE25_RECIST_baseline.pt"
        cfg_name = "efficienttam_s_512x512.yaml"
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if ckpt_path.name != filename:
                raise ValueError(f"eff-medsam2 only supports the small checkpoint: {filename}")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            local_ckpt_path = MEDSAM2_ROOT / "checkpoints" / filename
            ckpt_path = (
                local_ckpt_path
                if local_ckpt_path.exists()
                else hf_hub_download(
                    repo_id="wanglab/MedSAM2",
                    filename=filename,
                    cache_dir=str(MEDSAM2_ROOT / "checkpoints"),
                )
            )
        cfg_path = "/" + str(MEDSAM2_ROOT / "efficient_track_anything" / "configs" / cfg_name)
        device = args.device or "cpu"
        hydra_overrides = ["++model.compile_image_encoder=False"]
        predictor = build_efficienttam_video_predictor_npz(
            cfg_path,
            ckpt_path,
            device=device,
            hydra_overrides_extra=hydra_overrides,
        )
        name = "Efficient MedSAM2 small"

    return predictor, str(ckpt_path), name, device


def infer_medsam2(image: np.ndarray, recist: np.ndarray, spacing: np.ndarray, args) -> InferenceResult:
    import torch

    rng = np.random.RandomState(args.seed)
    prompt_specs = prompt_specs_from_recist(recist, spacing, args, rng, target="box")
    if not prompt_specs:
        raise ValueError("No prompts were provided")

    volume_uint8 = ensure_medsam2_uint8(image, args.intensity, args.window)
    frames, video_height, video_width = medsam2_preprocess(volume_uint8)

    with pushd(MEDSAM2_ROOT):
        predictor, ckpt_path, model_name, device = load_medsam2_predictor(args)

        segs = np.zeros(image.shape, dtype=np.uint16)
        boxes = []
        labels = [prompt.label for prompt in prompt_specs]

        for prompt in prompt_specs:
            z_min, z_max = medsam2_z_range_for_prompt(prompt)
            z_mid = prompt.z - z_min
            cropped_frames = frames[z_min : z_max + 1]

            with torch.inference_mode():
                state = predictor.init_state(cropped_frames, video_height, video_width)
                if prompt.kind == "box":
                    box_2d = prompt.box_xyxy.astype(np.float32)
                    boxes.append([box_2d[0], box_2d[1], prompt.z, box_2d[2], box_2d[3], prompt.z])
                    _, _, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=z_mid,
                        obj_id=1,
                        box=box_2d,
                    )
                elif prompt.kind == "points":
                    points = prompt.points_xy.astype(np.float32)
                    labels_arr = np.ones(len(points), dtype=np.int32)
                    _, _, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=z_mid,
                        obj_id=1,
                        points=points,
                        labels=labels_arr,
                    )
                else:
                    raise ValueError(f"Unsupported prompt kind: {prompt.kind}")

                mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
                _, _, masks = predictor.add_new_mask(state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
                segs[prompt.z, (masks[0] > 0.0).cpu().numpy()[0]] = prompt.label

                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=z_mid, reverse=False
                ):
                    segs[z_min + out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = prompt.label

                predictor.reset_state(state)
                state = predictor.init_state(cropped_frames, video_height, video_width)
                predictor.add_new_mask(state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)

                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=z_mid, reverse=True
                ):
                    segs[z_min + out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = prompt.label

                predictor.reset_state(state)

    boxes_array = np.asarray(boxes, dtype=np.float32).reshape((-1, 6)) if boxes else np.zeros((0, 6), dtype=np.float32)
    metadata = {
        "model": args.model,
        "model_name": model_name,
        "checkpoint": ckpt_path,
        "device": device,
        "prompt": "recist-box",
        "preprocessing": "medsam2_uint8_512_rgb_imagenet_norm",
        "intensity": args.intensity,
        "window": args.window,
        "spacing_zyx": spacing.tolist(),
        "labels": labels,
    }
    return InferenceResult(segs=segs, boxes_xyzxyz=boxes_array, metadata=metadata)


def resolve_pack_or_cwd_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PACK_ROOT / path).resolve()


def resolve_nninteractive_model_dir(args) -> str:
    if args.nninteractive_model_dir is not None:
        model_dir = resolve_pack_or_cwd_path(args.nninteractive_model_dir)
    else:
        model_dir = PACK_ROOT / "checkpoints" / "nnInteractive" / args.nninteractive_model_name

    required_files = ["dataset.json", "plans.json", "inference_session_class.json"]
    missing = [str(model_dir / name) for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "nnInteractive model folder is incomplete or missing. "
            f"Expected pack-local model at {model_dir}. Missing: {missing}"
        )

    if args.nninteractive_fold is None:
        folds = sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("fold_"))
        if len(folds) != 1:
            raise FileNotFoundError(
                "Could not infer nnInteractive fold. "
                f"Expected exactly one fold_* directory under {model_dir}, found {[p.name for p in folds]}. "
                "Pass --nninteractive-fold explicitly."
            )
        checkpoint_path = folds[0] / args.nninteractive_checkpoint
    else:
        fold_name = f"fold_{args.nninteractive_fold}"
        checkpoint_path = model_dir / fold_name / args.nninteractive_checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"nnInteractive checkpoint not found: {checkpoint_path}")

    return str(model_dir)


def infer_nninteractive(image: np.ndarray, recist: np.ndarray, spacing: np.ndarray, args) -> InferenceResult:
    import torch

    add_to_syspath(NNINTERACTIVE_ROOT)
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    rng = np.random.RandomState(args.seed)
    prompt_specs = prompt_specs_from_recist(recist, spacing, args, rng, target=args.nninteractive_prompt)
    if not prompt_specs:
        raise ValueError("No prompts were provided")

    model_dir = resolve_nninteractive_model_dir(args)
    device = args.device or "cuda:0"

    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=args.nninteractive_compile,
        verbose=args.verbose,
        torch_n_threads=args.torch_threads or os.cpu_count(),
        do_autozoom=not args.no_autozoom,
    )
    session.initialize_from_trained_model_folder(
        model_dir,
        use_fold=args.nninteractive_fold,
        checkpoint_name=args.nninteractive_checkpoint,
    )

    raw_image = np.asarray(image)
    session.set_image(raw_image[None])
    target = np.zeros(raw_image.shape, dtype=np.uint8)
    session.set_target_buffer(target)

    segs = np.zeros(raw_image.shape, dtype=np.uint16)
    boxes = []
    labels = [prompt.label for prompt in prompt_specs]

    for i, prompt in enumerate(prompt_specs):
        if i > 0:
            session.reset_interactions()

        prediction_center = None
        prediction_zoom_out_factor = None

        if prompt.kind == "box":
            import math as _math

            x1, y1, x2, y2 = prompt.box_xyxy.astype(int)
            spacing_z = float(spacing[0])
            spacing_y = float(spacing[1])
            spacing_x = float(spacing[2])
            diameter_mm = _math.hypot((int(y2) - int(y1)) * spacing_y, (int(x2) - int(x1)) * spacing_x)
            half_dz = max(10, int(round(diameter_mm / max(spacing_z, 0.1) / 2)))
            z_lo = max(0, int(prompt.z) - half_dz)
            z_hi = min(raw_image.shape[0], int(prompt.z) + half_dz + 1)
            bbox_dhw = [[z_lo, z_hi], [int(y1), int(y2) + 1], [int(x1), int(x2) + 1]]
            boxes.append([x1, y1, prompt.z, x2, y2, prompt.z])
            session.add_bbox_interaction(bbox_dhw, include_interaction=True, run_prediction=False)
            prediction_center = session.new_interaction_centers[-1]
            prediction_zoom_out_factor = session.new_interaction_zoom_out_factors[-1]
        elif prompt.kind == "points":
            if prompt.points_xy is not None and len(prompt.points_xy):
                cx, cy = np.mean(prompt.points_xy, axis=0)
                prediction_center = (int(prompt.z), int(round(float(cy))), int(round(float(cx))))
            for x, y in prompt.points_xy:
                session.add_point_interaction(
                    (prompt.z, int(round(y)), int(round(x))),
                    include_interaction=True,
                    run_prediction=False,
                )
            if prompt.negative_points_xy is not None:
                for x, y in prompt.negative_points_xy:
                    session.add_point_interaction(
                        (prompt.z, int(round(y)), int(round(x))),
                        include_interaction=False,
                        run_prediction=False,
                    )
        else:
            raise ValueError(f"Unsupported prompt kind: {prompt.kind}")

        # Predict around a foreground/lesion center. Negative clicks are added
        # after foreground clicks, so the last interaction center can be bg.
        if prediction_center is not None:
            session.new_interaction_centers = [prediction_center]
        else:
            session.new_interaction_centers = [session.new_interaction_centers[-1]]
        if prediction_zoom_out_factor is not None:
            session.new_interaction_zoom_out_factors = [prediction_zoom_out_factor]
        else:
            session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
        session._predict()
        segs[target > 0] = prompt.label

    boxes_array = np.asarray(boxes, dtype=np.float32).reshape((-1, 6)) if boxes else np.zeros((0, 6), dtype=np.float32)
    metadata = {
        "model": args.model,
        "model_dir": model_dir,
        "device": device,
        "prompt": f"recist-{args.nninteractive_prompt.replace('_', '-')}",
        "preprocessing": "nninteractive_internal_raw_set_image",
        "spacing_zyx": spacing.tolist(),
        "labels": labels,
        "nninteractive_checkpoint": args.nninteractive_checkpoint,
        "nninteractive_fold": args.nninteractive_fold,
    }
    return InferenceResult(segs=segs, boxes_xyzxyz=boxes_array, metadata=metadata)


def save_outputs(
    output_path: Path,
    result: InferenceResult,
    image: LoadedImage,
    recist: np.ndarray,
    recist_source: str,
    args,
    duration_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "image_path": str(Path(args.image).resolve()),
            "image_key": image.source_key,
            "recist_source": recist_source,
            "image_shape_dhw": list(image.array.shape),
            "duration_s": duration_s,
        }
    )
    np.savez_compressed(
        output_path,
        segs=result.segs,
        recist=recist,
        boxes=result.boxes_xyzxyz,
        spacing=image.spacing,
        metadata=json.dumps(metadata, indent=2, sort_keys=True),
    )

    if args.output_nifti is not None:
        write_nifti(Path(args.output_nifti), result.segs, image)


def write_nifti(path: Path, segs: np.ndarray, image: LoadedImage) -> None:
    import SimpleITK as sitk

    path.parent.mkdir(parents=True, exist_ok=True)
    seg_img = sitk.GetImageFromArray(segs.astype(np.uint16))
    if image.sitk_image is not None:
        seg_img.CopyInformation(image.sitk_image)
    else:
        # SimpleITK spacing is x,y,z while this script stores z,y,x.
        seg_img.SetSpacing((float(image.spacing[2]), float(image.spacing[1]), float(image.spacing[0])))
    sitk.WriteImage(seg_img, str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RECIST-prompted segmentation with MedSAM2, Efficient MedSAM2, or nnInteractive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Input image path: NPZ/NPY/NIfTI/PNG/JPG. NPZ defaults to key 'imgs'.")
    parser.add_argument("--output", required=True, help="Output .npz path containing segs, recist, boxes, spacing, metadata.")
    parser.add_argument("--model", required=True, choices=("eff-medsam2", "medsam2", "nninteractive"))
    parser.add_argument("--recist", help="Optional RECIST mask path: NPZ/NPY/NIfTI/PNG. Same shape as image.")
    parser.add_argument(
        "--recist-line",
        action="append",
        default=[],
        help="RECIST line as 'z,x1,y1,x2,y2' or 'z,x1,y1,x2,y2,label'. Can be repeated.",
    )
    parser.add_argument("--image-key", help="NPZ key for --image.")
    parser.add_argument("--recist-key", help="NPZ key for --recist, or input NPZ recist key.")
    parser.add_argument(
        "--recist-space",
        choices=("strict", "index"),
        default="strict",
        help=(
            "How to interpret NIfTI --recist. 'strict' requires matching NIfTI "
            "size/spacing/origin/direction; 'index' ignores NIfTI geometry and "
            "keeps legacy array-only behavior."
        ),
    )
    parser.add_argument("--spacing", help="Override spacing as 'z,y,x'.")
    parser.add_argument("--output-nifti", help="Optional output segmentation NIfTI path.")
    parser.add_argument("--shift", type=int, default=0, help="Extra pixels added around RECIST-derived bbox.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", help="Torch device. Defaults: medsam2=cuda, eff-medsam2=cpu, nninteractive=cuda:0.")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Optional local MedSAM2/Efficient MedSAM2 checkpoint path. "
            "By default the pack-local RECIST checkpoints are used. "
            "eff-medsam2 accepts only eff_medsam2_small_FLARE25_RECIST_baseline.pt."
        ),
    )
    parser.add_argument(
        "--intensity",
        choices=("preserve", "minmax", "window"),
        default="preserve",
        help="MedSAM2-only conversion to uint8 [0,255]. nnInteractive always receives the raw image.",
    )
    parser.add_argument("--window", help="MedSAM2-only intensity window as 'MIN,MAX' when --intensity window is used.")
    parser.add_argument(
        "--nninteractive-model-dir",
        help="Optional nnInteractive model folder. Relative paths are resolved from cwd first, then the pack root.",
    )
    parser.add_argument(
        "--nninteractive-model-name",
        default="nnInteractive_v1.0",
        help="Pack-local folder name under checkpoints/nnInteractive when model dir is omitted.",
    )
    parser.add_argument(
        "--nninteractive-fold",
        default=None,
        help="Fold passed to nnInteractive. Defaults to inferring the only fold_* folder in the model dir.",
    )
    parser.add_argument("--nninteractive-checkpoint", default="checkpoint_final.pth", help="Checkpoint filename for nnInteractive.")
    parser.add_argument(
        "--nninteractive-prompt",
        choices=("5_points", "5pos_4neg", "5pos_6neg"),
        default="5_points",
        help=(
            "RECIST-derived nnInteractive prompt. 5_points uses five foreground points on the RECIST line. "
            "5pos_4neg adds four RECIST-derived rotated-box corner background points. "
            "5pos_6neg also adds two minor-axis background points."
        ),
    )
    parser.add_argument("--nninteractive-compile", action="store_true", help="Enable torch.compile for nnInteractive.")
    parser.add_argument("--no-autozoom", action="store_true", help="Disable nnInteractive autozoom.")
    parser.add_argument("--torch-threads", type=int, help="CPU thread count for nnInteractive preprocessing.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)

    start = time.time()
    loaded = load_image(image_path, args.image_key, args.spacing)
    recist, recist_source = load_recist(
        image_path=image_path,
        recist_path=Path(args.recist) if args.recist else None,
        recist_key=args.recist_key,
        recist_lines=args.recist_line,
        shape=loaded.array.shape,
        reference_sitk_image=loaded.sitk_image,
        recist_space=args.recist_space,
    )

    if args.model in {"eff-medsam2", "medsam2"}:
        result = infer_medsam2(loaded.array, recist, loaded.spacing, args)
    else:
        result = infer_nninteractive(loaded.array, recist, loaded.spacing, args)

    duration_s = time.time() - start
    save_outputs(output_path, result, loaded, recist, recist_source, args, duration_s)
    print(f"wrote {output_path}")
    if args.output_nifti:
        print(f"wrote {args.output_nifti}")
    print(json.dumps(result.metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
