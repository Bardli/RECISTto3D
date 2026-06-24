#!/usr/bin/env python3
"""Run the three RECISTto3D model backends concurrently from Python.

Primary use:

    from run_three_models_parallel import load_all_models, infer_with_loaded_models

    models = load_all_models(device="cuda:0")
    results = infer_with_loaded_models(
        models=models,
        image="image.nii.gz",
        recist="recist.nii.gz",
        intensity="window",
        window="-175,275",
    )

The implementation imports and calls ``recist_infer.py`` functions directly in
three Python worker processes. It does not shell out to ``recist_infer.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
MODELS = ("eff-medsam2", "medsam2", "nninteractive")


@dataclass(frozen=True)
class ModelResult:
    model: str
    output_nifti: str
    duration_s: float
    metadata: dict


@dataclass
class LoadedModel:
    model: str
    handle: object
    device: str
    metadata: dict


@dataclass
class LoadedModels:
    eff_medsam2: LoadedModel
    medsam2: LoadedModel
    nninteractive: LoadedModel
    metadata: dict[str, dict]


@dataclass(frozen=True)
class _ModelConfig:
    model: str
    image: str
    output_nifti: str
    recist: str | None
    recist_lines: tuple[str, ...]
    image_key: str | None
    recist_key: str | None
    recist_space: str
    spacing: str | None
    shift: int
    seed: int
    intensity: str
    window: str | None
    device: str | None
    checkpoint: str | None
    nninteractive_model_dir: str | None
    nninteractive_model_name: str
    nninteractive_fold: str | None
    nninteractive_checkpoint: str
    nninteractive_prompt: str
    nninteractive_compile: bool
    no_autozoom: bool
    torch_threads: int | None
    verbose: bool


# --- Self-contained RECIST-from-GT extraction -------------------------------
# Turns a GT label mask into the app's RECIST text format, one line per lesion:
#   "z,x1,y1,x2,y2,label"  (axial voxel indices, label == connected-component id)
# Same canonical algorithm as the rest of the pipeline (cc3d-26 instances, drop
# tiny lesions, key slice = max axial area, longest external-contour pair), but
# intentionally inlined here so this helper depends ONLY on third-party libs and
# never imports another module in this repo.

_MIN_LESION_VOXELS = 10        # drop lesions smaller than this (benchmark convention)
_RECIST_MAX_CONTOUR_PTS = 500  # subsample contour to at most this many points before pdist


def recist_lines_from_gt_mask(
    gt_mask: "str | Path | object",
    *,
    min_voxels: int = _MIN_LESION_VOXELS,
    expected_shape: "tuple[int, int, int] | None" = None,
) -> list[str]:
    """Extract one RECIST diameter line per GT lesion as app-ready text.

    Returns a list of ``"z,x1,y1,x2,y2,label"`` strings (axial voxel indices,
    ``label`` == connected-component instance id), a drop-in for the app's
    ``recist-line-box`` textbox and ``_validate_recist_lines``. Returns ``[]``
    when no lesion qualifies. Fully self-contained: depends only on numpy /
    SimpleITK / scipy / (cv2 or skimage), never on another module in this repo.

    ``gt_mask`` may be a NIfTI/.npy/.npz path or a 3D ``(z, y, x)`` ndarray.
    ``expected_shape`` optionally guards that the mask matches the loaded image.
    """
    import numpy as np

    gt = _as_mask_array(gt_mask)
    if expected_shape is not None and gt.shape != tuple(expected_shape):
        raise ValueError(f"GT mask shape {gt.shape} != image shape {tuple(expected_shape)}")

    instance = _connected_components_26(gt > 0)
    lines: list[str] = []
    for lid in (int(v) for v in np.unique(instance) if v != 0):
        lesion = instance == lid
        if int(lesion.sum()) < min_voxels:
            continue
        z = int(np.argmax(lesion.sum(axis=(1, 2))))  # key slice = max axial area
        endpoints = _longest_diameter_xy(lesion[z].astype(np.uint8))
        if endpoints is None:
            continue
        (x1, y1), (x2, y2) = endpoints
        lines.append(f"{z},{int(x1)},{int(y1)},{int(x2)},{int(y2)},{lid}")
    return lines


def _as_mask_array(gt_mask: "str | Path | object"):
    """Load a GT mask into a 3D ``(z, y, x)`` integer ndarray (self-contained)."""
    import numpy as np

    if isinstance(gt_mask, np.ndarray):
        arr = gt_mask
    else:
        p = str(gt_mask)
        lower = p.lower()
        if lower.endswith((".nii", ".nii.gz")):
            import SimpleITK as sitk

            arr = sitk.GetArrayFromImage(sitk.ReadImage(p))  # -> (z, y, x)
        elif lower.endswith(".npz"):
            with np.load(p, allow_pickle=True) as data:
                key = next((k for k in ("gts", "mask", "recist", "arr_0") if k in data), None)
                if key is None:
                    raise ValueError(f"{p} has no gts/mask/recist/arr_0 array")
                arr = data[key]
        elif lower.endswith(".npy"):
            arr = np.load(p, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported GT mask format: {p}")
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"GT mask must be 3D (z, y, x); got shape {arr.shape}")
    return arr


def _connected_components_26(binary):
    """26-connectivity instance labeling; cc3d if available, else scipy."""
    import numpy as np

    try:
        import cc3d

        return cc3d.connected_components(binary.astype(np.uint8), connectivity=26)
    except ImportError:
        from scipy import ndimage

        structure = np.ones((3, 3, 3), dtype=int)  # full 26-connectivity
        instance, _ = ndimage.label(binary.astype(np.uint8), structure=structure)
        return instance


def _longest_diameter_xy(mask_2d):
    """Farthest external-contour point pair as ``((x1, y1), (x2, y2))`` or None."""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    try:
        import cv2

        contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        pts = np.vstack(contours).squeeze()  # (x, y)
    except ImportError:
        from skimage import measure

        contours_yx = measure.find_contours(mask_2d, 0.5)
        if not contours_yx:
            return None
        yx = np.concatenate(contours_yx, axis=0)
        pts = np.stack([yx[:, 1], yx[:, 0]], axis=1)  # (row, col) -> (x, y)

    if pts.ndim != 2 or len(pts) < 2:
        return None
    if len(pts) > _RECIST_MAX_CONTOUR_PTS:
        pts = pts[np.linspace(0, len(pts) - 1, _RECIST_MAX_CONTOUR_PTS, dtype=int)]
    dist_matrix = squareform(pdist(pts))
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    return np.rint(pts[i]).astype(int), np.rint(pts[j]).astype(int)


def _strip_known_suffixes(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".npz", ".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _as_path_string(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def _resolve_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _model_device(
    model: str,
    common_device: str | None,
    eff_medsam2_device: str | None,
    medsam2_device: str | None,
    nninteractive_device: str | None,
) -> str | None:
    if model == "eff-medsam2":
        return eff_medsam2_device or common_device
    if model == "medsam2":
        return medsam2_device or common_device
    if model == "nninteractive":
        return nninteractive_device or common_device
    raise ValueError(f"unknown model: {model}")


def _model_checkpoint(
    model: str,
    eff_medsam2_checkpoint: str | Path | None,
    medsam2_checkpoint: str | Path | None,
) -> str | None:
    if model == "eff-medsam2":
        return _as_path_string(eff_medsam2_checkpoint)
    if model == "medsam2":
        return _as_path_string(medsam2_checkpoint)
    if model == "nninteractive":
        return None
    raise ValueError(f"unknown model: {model}")


def _namespace_from_config(config: _ModelConfig) -> Namespace:
    return Namespace(
        image=config.image,
        model=config.model,
        recist=config.recist,
        recist_line=list(config.recist_lines),
        image_key=config.image_key,
        recist_key=config.recist_key,
        recist_space=config.recist_space,
        spacing=config.spacing,
        output_nifti=config.output_nifti,
        shift=config.shift,
        seed=config.seed,
        device=config.device,
        checkpoint=config.checkpoint,
        intensity=config.intensity,
        window=config.window,
        nninteractive_model_dir=config.nninteractive_model_dir,
        nninteractive_model_name=config.nninteractive_model_name,
        nninteractive_fold=config.nninteractive_fold,
        nninteractive_checkpoint=config.nninteractive_checkpoint,
        nninteractive_prompt=config.nninteractive_prompt,
        nninteractive_compile=config.nninteractive_compile,
        no_autozoom=config.no_autozoom,
        torch_threads=config.torch_threads,
        verbose=config.verbose,
    )


def _normalize_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device == "gpu":
        return "cuda:0"
    return device


def _model_loader_args(
    *,
    model: str,
    device: str | None,
    checkpoint: str | Path | None = None,
    nninteractive_model_dir: str | Path | None = None,
    nninteractive_model_name: str = "nnInteractive_v1.0",
    nninteractive_fold: str | None = None,
    nninteractive_checkpoint: str = "checkpoint_final.pth",
    nninteractive_compile: bool = False,
    no_autozoom: bool = False,
    torch_threads: int | None = None,
    verbose: bool = False,
) -> Namespace:
    return Namespace(
        model=model,
        device=_normalize_device(device),
        checkpoint=_as_path_string(checkpoint),
        nninteractive_model_dir=_as_path_string(nninteractive_model_dir),
        nninteractive_model_name=nninteractive_model_name,
        nninteractive_fold=nninteractive_fold,
        nninteractive_checkpoint=nninteractive_checkpoint,
        nninteractive_compile=nninteractive_compile,
        no_autozoom=no_autozoom,
        torch_threads=torch_threads,
        verbose=verbose,
    )


def _load_medsam2_model(args: Namespace) -> LoadedModel:
    from recist_infer import MEDSAM2_ROOT, load_medsam2_predictor, pushd

    with pushd(MEDSAM2_ROOT):
        predictor, checkpoint, model_name, device = load_medsam2_predictor(args)

    metadata = {
        "model": args.model,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "device": device,
    }
    return LoadedModel(model=args.model, handle=predictor, device=str(device), metadata=metadata)


def _load_nninteractive_model(args: Namespace) -> LoadedModel:
    import torch
    from recist_infer import NNINTERACTIVE_ROOT, add_to_syspath, resolve_nninteractive_model_dir

    add_to_syspath(NNINTERACTIVE_ROOT)
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

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

    metadata = {
        "model": "nninteractive",
        "model_dir": model_dir,
        "device": device,
        "nninteractive_checkpoint": args.nninteractive_checkpoint,
        "nninteractive_fold": args.nninteractive_fold,
    }
    return LoadedModel(model="nninteractive", handle=session, device=str(device), metadata=metadata)


def load_all_models(
    *,
    device: str | None = "cuda:0",
    eff_medsam2_device: str | None = None,
    medsam2_device: str | None = None,
    nninteractive_device: str | None = None,
    eff_medsam2_checkpoint: str | Path | None = None,
    medsam2_checkpoint: str | Path | None = None,
    nninteractive_model_dir: str | Path | None = None,
    nninteractive_model_name: str = "nnInteractive_v1.0",
    nninteractive_fold: str | None = None,
    nninteractive_checkpoint: str = "checkpoint_final.pth",
    nninteractive_compile: bool = False,
    no_autozoom: bool = False,
    torch_threads: int | None = None,
    verbose: bool = False,
) -> LoadedModels:
    """Load eff-medsam2, medsam2, and nninteractive into this Python process.

    Pass ``device="cpu"`` to keep all models on CPU, ``device="cuda:0"`` to
    place them on one GPU, or use the per-model device arguments to split models
    across devices. ``device="gpu"`` is accepted as an alias for ``"cuda:0"``.
    """

    eff_args = _model_loader_args(
        model="eff-medsam2",
        device=eff_medsam2_device or device,
        checkpoint=eff_medsam2_checkpoint,
    )
    medsam_args = _model_loader_args(
        model="medsam2",
        device=medsam2_device or device,
        checkpoint=medsam2_checkpoint,
    )
    nninteractive_args = _model_loader_args(
        model="nninteractive",
        device=nninteractive_device or device,
        nninteractive_model_dir=nninteractive_model_dir,
        nninteractive_model_name=nninteractive_model_name,
        nninteractive_fold=nninteractive_fold,
        nninteractive_checkpoint=nninteractive_checkpoint,
        nninteractive_compile=nninteractive_compile,
        no_autozoom=no_autozoom,
        torch_threads=torch_threads,
        verbose=verbose,
    )

    eff_medsam2 = _load_medsam2_model(eff_args)
    medsam2 = _load_medsam2_model(medsam_args)
    nninteractive = _load_nninteractive_model(nninteractive_args)
    metadata = {
        "eff_medsam2": eff_medsam2.metadata,
        "medsam2": medsam2.metadata,
        "nninteractive": nninteractive.metadata,
    }
    return LoadedModels(
        eff_medsam2=eff_medsam2,
        medsam2=medsam2,
        nninteractive=nninteractive,
        metadata=metadata,
    )



# Gradio insertion point:
# After ``infer_with_loaded_models(...)`` returns three model-specific NIfTI
# masks, call ``make_bitmask_labels_image(...)`` to create one display mask and
# return ``_file_url(combined_mask_path)`` through app.py's existing
# ``mask_url_state`` output. The current frontend can keep calling
# ``window.recistTo3DViewer.loadMask(maskUrl)``, because ``loadMask`` already
# uses NiiVue's ``nv.loadDrawingFromUrl(maskUrl, false)``.
#
# The combined NIfTI is a single 3D label image with bit-encoded labels:
#   0 = background
#   1 = eff-medsam2 only
#   2 = medsam2 only
#   4 = nninteractive only
#   3 = eff-medsam2 + medsam2 overlap
#   5 = eff-medsam2 + nninteractive overlap
#   6 = medsam2 + nninteractive overlap
#   7 = all three overlap
#
# In app.py, add a Gradio CheckboxGroup for model visibility and a JavaScript
# method that toggles NiiVue colormap alpha for labels whose bits match the
# selected masks. This preserves overlap information while still loading just
# one NIfTI into the existing viewer.
def make_bitmask_labels_image(
    eff_medsam2_mask: str | Path,
    medsam2_mask: str | Path,
    nninteractive_mask: str | Path,
    output_nifti: str | Path,
    *,
    reference_nifti: str | Path | None = None,
    require_same_geometry: bool = True,
) -> str:
    """Write one bit-encoded display NIfTI from three binary/model masks."""

    import numpy as np
    import SimpleITK as sitk

    mask_paths = {
        "eff-medsam2": Path(eff_medsam2_mask),
        "medsam2": Path(medsam2_mask),
        "nninteractive": Path(nninteractive_mask),
    }
    images = {name: sitk.ReadImage(str(path)) for name, path in mask_paths.items()}
    reference_image = sitk.ReadImage(str(reference_nifti)) if reference_nifti is not None else images["eff-medsam2"]

    def geometry_tuple(image) -> tuple:
        return (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())

    if require_same_geometry:
        reference_geometry = geometry_tuple(reference_image)
        mismatches = [
            name
            for name, image in images.items()
            if geometry_tuple(image) != reference_geometry
        ]
        if mismatches:
            raise ValueError(
                "All masks must share the same NIfTI geometry as the reference image. "
                f"Mismatched masks: {mismatches}"
            )

    arrays = {name: sitk.GetArrayFromImage(image) > 0 for name, image in images.items()}
    shape = arrays["eff-medsam2"].shape
    shape_mismatches = [name for name, array in arrays.items() if array.shape != shape]
    if shape_mismatches:
        raise ValueError(f"All masks must have the same array shape. Mismatched masks: {shape_mismatches}")

    combined = np.zeros(shape, dtype=np.uint8)
    combined[arrays["eff-medsam2"]] |= 1
    combined[arrays["medsam2"]] |= 2
    combined[arrays["nninteractive"]] |= 4

    output_path = Path(output_nifti)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = sitk.GetImageFromArray(combined)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, str(output_path))
    return str(output_path)




def _loaded_model_for_name(models: LoadedModels, model: str) -> LoadedModel:
    if model == "eff-medsam2":
        return models.eff_medsam2
    if model == "medsam2":
        return models.medsam2
    if model == "nninteractive":
        return models.nninteractive
    raise ValueError(f"unknown model: {model}")


def _infer_medsam2_with_loaded_model(image, recist, spacing, args: Namespace, loaded_model: LoadedModel):
    import numpy as np
    from recist_infer import InferenceResult, run_medsam2_loop

    rng = np.random.RandomState(args.seed)
    predictor = loaded_model.handle
    segs, boxes_array, labels = run_medsam2_loop(
        image, recist, spacing, args, predictor, loaded_model.device, rng=rng
    )

    metadata = dict(loaded_model.metadata)
    metadata.update(
        {
            "model": args.model,
            "prompt": "recist-box",
            "preprocessing": "medsam2_uint8_512_rgb_imagenet_norm",
            "intensity": args.intensity,
            "window": args.window,
            "spacing_zyx": spacing.tolist(),
            "labels": labels,
            "loaded_model_reused": True,
        }
    )
    return InferenceResult(segs=segs, boxes_xyzxyz=boxes_array, metadata=metadata)


def _infer_nninteractive_with_loaded_model(image, recist, spacing, args: Namespace, loaded_model: LoadedModel):
    import math as _math
    import numpy as np
    from recist_infer import InferenceResult, prompt_specs_from_recist

    rng = np.random.RandomState(args.seed)
    prompt_specs = prompt_specs_from_recist(recist, spacing, args, rng, target=args.nninteractive_prompt)
    if not prompt_specs:
        raise ValueError("No prompts were provided")

    session = loaded_model.handle
    if hasattr(session, "reset_interactions"):
        session.reset_interactions()

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
    metadata = dict(loaded_model.metadata)
    metadata.update(
        {
            "model": args.model,
            "prompt": f"recist-{args.nninteractive_prompt.replace('_', '-')}",
            "preprocessing": "nninteractive_internal_raw_set_image",
            "spacing_zyx": spacing.tolist(),
            "labels": labels,
            "loaded_model_reused": True,
        }
    )
    return InferenceResult(segs=segs, boxes_xyzxyz=boxes_array, metadata=metadata)


def _run_one_loaded_model(config: _ModelConfig, loaded_model: LoadedModel, loaded_image, recist, recist_source: str) -> ModelResult:
    from recist_infer import write_nifti

    args = _namespace_from_config(config)
    output_nifti_path = Path(args.output_nifti)

    start = time.time()
    if args.model in {"eff-medsam2", "medsam2"}:
        result = _infer_medsam2_with_loaded_model(loaded_image.array, recist, loaded_image.spacing, args, loaded_model)
    elif args.model == "nninteractive":
        result = _infer_nninteractive_with_loaded_model(loaded_image.array, recist, loaded_image.spacing, args, loaded_model)
    else:
        raise ValueError(f"unknown model: {args.model}")

    duration_s = time.time() - start
    metadata = dict(result.metadata)
    metadata.update(
        {
            "image_path": str(Path(args.image).resolve()),
            "image_key": loaded_image.source_key,
            "recist_source": recist_source,
            "image_shape_dhw": list(loaded_image.array.shape),
            "duration_s": duration_s,
            "output_format": "nifti",
        }
    )
    result.metadata = metadata
    write_nifti(output_nifti_path, result.segs, loaded_image)
    return ModelResult(
        model=args.model,
        output_nifti=str(output_nifti_path),
        duration_s=duration_s,
        metadata=metadata,
    )


def infer_with_loaded_models(
    *,
    models: LoadedModels,
    image: str | Path,
    recist: str | Path | None = None,
    recist_lines: Sequence[str] = (),
    output_dir: str | Path = "outputs/three_models",
    output_prefix: str | None = None,
    image_key: str | None = None,
    recist_key: str | None = None,
    recist_space: str = "strict",
    spacing: str | None = None,
    shift: int = 0,
    seed: int = 2024,
    intensity: str = "preserve",
    window: str | None = None,
    nninteractive_prompt: str = "5_points",
    verbose: bool = False,
    run_concurrent: bool = False,
) -> list[ModelResult]:
    """Run inference with models already returned by ``load_all_models``.

    Loaded model objects stay in this process. The default is sequential
    execution because repeated threaded calls can retain per-thread runtime
    caches and raise process RSS in Gradio-style long-running services.
    """

    configs = _build_configs(
        image=image,
        recist=recist,
        recist_lines=recist_lines,
        output_dir=output_dir,
        output_prefix=output_prefix,
        image_key=image_key,
        recist_key=recist_key,
        recist_space=recist_space,
        spacing=spacing,
        shift=shift,
        seed=seed,
        intensity=intensity,
        window=window,
        device=None,
        eff_medsam2_device=None,
        medsam2_device=None,
        nninteractive_device=None,
        eff_medsam2_checkpoint=None,
        medsam2_checkpoint=None,
        nninteractive_model_dir=None,
        nninteractive_model_name="nnInteractive_v1.0",
        nninteractive_fold=None,
        nninteractive_checkpoint="checkpoint_final.pth",
        nninteractive_prompt=nninteractive_prompt,
        nninteractive_compile=False,
        no_autozoom=False,
        torch_threads=None,
        verbose=verbose,
    )

    from recist_infer import load_image, load_recist

    image_path = Path(_as_path_string(image) or "")
    loaded_image = load_image(image_path, image_key, spacing)
    recist_array, recist_source = load_recist(
        image_path=image_path,
        recist_path=Path(recist) if recist is not None else None,
        recist_key=recist_key,
        recist_lines=list(recist_lines),
        shape=loaded_image.array.shape,
        reference_sitk_image=loaded_image.sitk_image,
        recist_space=recist_space,
    )

    if not run_concurrent:
        return [
            _run_one_loaded_model(config, _loaded_model_for_name(models, config.model), loaded_image, recist_array, recist_source)
            for config in configs
        ]

    results_by_model: dict[str, ModelResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        future_to_model = {
            executor.submit(
                _run_one_loaded_model,
                config,
                _loaded_model_for_name(models, config.model),
                loaded_image,
                recist_array,
                recist_source,
            ): config.model
            for config in configs
        }
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            results_by_model[model] = future.result()

    return [results_by_model[model] for model in MODELS]




def _run_one_model(config: _ModelConfig) -> ModelResult:
    from recist_infer import infer_medsam2, infer_nninteractive, load_image, load_recist, write_nifti

    args = _namespace_from_config(config)
    image_path = Path(args.image)
    output_nifti_path = Path(args.output_nifti)

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
    elif args.model == "nninteractive":
        result = infer_nninteractive(loaded.array, recist, loaded.spacing, args)
    else:
        raise ValueError(f"unknown model: {args.model}")

    duration_s = time.time() - start
    metadata = dict(result.metadata)
    metadata.update(
        {
            "image_path": str(Path(args.image).resolve()),
            "image_key": loaded.source_key,
            "recist_source": recist_source,
            "image_shape_dhw": list(loaded.array.shape),
            "duration_s": duration_s,
            "output_format": "nifti",
        }
    )
    result.metadata = metadata
    write_nifti(output_nifti_path, result.segs, loaded)
    return ModelResult(
        model=args.model,
        output_nifti=str(output_nifti_path),
        duration_s=duration_s,
        metadata=metadata,
    )


def _build_configs(
    *,
    image: str | Path,
    recist: str | Path | None,
    recist_lines: Sequence[str],
    output_dir: str | Path,
    output_prefix: str | None,
    image_key: str | None,
    recist_key: str | None,
    recist_space: str,
    spacing: str | None,
    shift: int,
    seed: int,
    intensity: str,
    window: str | None,
    device: str | None,
    eff_medsam2_device: str | None,
    medsam2_device: str | None,
    nninteractive_device: str | None,
    eff_medsam2_checkpoint: str | Path | None,
    medsam2_checkpoint: str | Path | None,
    nninteractive_model_dir: str | Path | None,
    nninteractive_model_name: str,
    nninteractive_fold: str | None,
    nninteractive_checkpoint: str,
    nninteractive_prompt: str,
    nninteractive_compile: bool,
    no_autozoom: bool,
    torch_threads: int | None,
    verbose: bool,
) -> list[_ModelConfig]:
    output_root = _resolve_output_dir(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image_str = _as_path_string(image)
    if image_str is None:
        raise ValueError("image is required")
    recist_str = _as_path_string(recist)
    prefix = output_prefix or _strip_known_suffixes(Path(image_str))

    configs = []
    for model in MODELS:
        safe_model = model.replace("-", "_")
        output_nii = output_root / f"{prefix}_{safe_model}.nii.gz"
        configs.append(
            _ModelConfig(
                model=model,
                image=image_str,
                output_nifti=str(output_nii),
                recist=recist_str,
                recist_lines=tuple(recist_lines),
                image_key=image_key,
                recist_key=recist_key,
                recist_space=recist_space,
                spacing=spacing,
                shift=shift,
                seed=seed,
                intensity=intensity,
                window=window,
                device=_model_device(model, device, eff_medsam2_device, medsam2_device, nninteractive_device),
                checkpoint=_model_checkpoint(model, eff_medsam2_checkpoint, medsam2_checkpoint),
                nninteractive_model_dir=_as_path_string(nninteractive_model_dir),
                nninteractive_model_name=nninteractive_model_name,
                nninteractive_fold=nninteractive_fold,
                nninteractive_checkpoint=nninteractive_checkpoint,
                nninteractive_prompt=nninteractive_prompt,
                nninteractive_compile=nninteractive_compile,
                no_autozoom=no_autozoom,
                torch_threads=torch_threads,
                verbose=verbose,
            )
        )
    return configs


def run_three_models(
    *,
    image: str | Path,
    loaded_models: LoadedModels | None = None,
    recist: str | Path | None = None,
    recist_lines: Sequence[str] = (),
    output_dir: str | Path = "outputs/three_models",
    output_prefix: str | None = None,
    image_key: str | None = None,
    recist_key: str | None = None,
    recist_space: str = "strict",
    spacing: str | None = None,
    shift: int = 0,
    seed: int = 2024,
    intensity: str = "preserve",
    window: str | None = None,
    device: str | None = None,
    eff_medsam2_device: str | None = None,
    medsam2_device: str | None = None,
    nninteractive_device: str | None = None,
    eff_medsam2_checkpoint: str | Path | None = None,
    medsam2_checkpoint: str | Path | None = None,
    nninteractive_model_dir: str | Path | None = None,
    nninteractive_model_name: str = "nnInteractive_v1.0",
    nninteractive_fold: str | None = None,
    nninteractive_checkpoint: str = "checkpoint_final.pth",
    nninteractive_prompt: str = "5_points",
    nninteractive_compile: bool = False,
    no_autozoom: bool = False,
    torch_threads: int | None = None,
    verbose: bool = False,
    start_method: str | None = None,
) -> list[ModelResult]:
    """Run eff-medsam2, medsam2, and nninteractive concurrently.

    If ``loaded_models`` is provided, the already-loaded model handles are reused
    in this process. Otherwise, this function keeps the older process-based path
    where each worker loads its own model.
    """

    if loaded_models is not None:
        return infer_with_loaded_models(
            models=loaded_models,
            image=image,
            recist=recist,
            recist_lines=recist_lines,
            output_dir=output_dir,
            output_prefix=output_prefix,
                image_key=image_key,
            recist_key=recist_key,
            recist_space=recist_space,
            spacing=spacing,
            shift=shift,
            seed=seed,
            intensity=intensity,
            window=window,
            nninteractive_prompt=nninteractive_prompt,
            verbose=verbose,
        )

    configs = _build_configs(
        image=image,
        recist=recist,
        recist_lines=recist_lines,
        output_dir=output_dir,
        output_prefix=output_prefix,
        image_key=image_key,
        recist_key=recist_key,
        recist_space=recist_space,
        spacing=spacing,
        shift=shift,
        seed=seed,
        intensity=intensity,
        window=window,
        device=device,
        eff_medsam2_device=eff_medsam2_device,
        medsam2_device=medsam2_device,
        nninteractive_device=nninteractive_device,
        eff_medsam2_checkpoint=eff_medsam2_checkpoint,
        medsam2_checkpoint=medsam2_checkpoint,
        nninteractive_model_dir=nninteractive_model_dir,
        nninteractive_model_name=nninteractive_model_name,
        nninteractive_fold=nninteractive_fold,
        nninteractive_checkpoint=nninteractive_checkpoint,
        nninteractive_prompt=nninteractive_prompt,
        nninteractive_compile=nninteractive_compile,
        no_autozoom=no_autozoom,
        torch_threads=torch_threads,
        verbose=verbose,
    )

    context = mp.get_context(start_method) if start_method is not None else None
    executor_kwargs = {"max_workers": len(configs)}
    if context is not None:
        executor_kwargs["mp_context"] = context

    results_by_model: dict[str, ModelResult] = {}
    with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
        future_to_model = {executor.submit(_run_one_model, config): config.model for config in configs}
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            results_by_model[model] = future.result()

    return [results_by_model[model] for model in MODELS]


def results_to_dicts(results: Iterable[ModelResult]) -> list[dict]:
    return [asdict(result) for result in results]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run eff-medsam2, medsam2, and nninteractive concurrently via Python function calls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--recist")
    parser.add_argument("--recist-line", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/three_models")
    parser.add_argument("--output-prefix")
    parser.add_argument("--image-key")
    parser.add_argument("--recist-key")
    parser.add_argument("--recist-space", choices=("strict", "index"), default="strict")
    parser.add_argument("--spacing")
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--intensity", choices=("preserve", "minmax", "window"), default="preserve")
    parser.add_argument("--window")
    parser.add_argument("--device")
    parser.add_argument("--eff-medsam2-device")
    parser.add_argument("--medsam2-device")
    parser.add_argument("--nninteractive-device")
    parser.add_argument("--eff-medsam2-checkpoint")
    parser.add_argument("--medsam2-checkpoint")
    parser.add_argument("--nninteractive-model-dir")
    parser.add_argument("--nninteractive-model-name", default="nnInteractive_v1.0")
    parser.add_argument("--nninteractive-fold")
    parser.add_argument("--nninteractive-checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--nninteractive-prompt", choices=("5_points", "5pos_4neg", "5pos_6neg"), default="5_points")
    parser.add_argument("--nninteractive-compile", action="store_true")
    parser.add_argument("--no-autozoom", action="store_true")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--start-method", choices=tuple(mp.get_all_start_methods()))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = run_three_models(
        image=args.image,
        recist=args.recist,
        recist_lines=args.recist_line,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        image_key=args.image_key,
        recist_key=args.recist_key,
        recist_space=args.recist_space,
        spacing=args.spacing,
        shift=args.shift,
        seed=args.seed,
        intensity=args.intensity,
        window=args.window,
        device=args.device,
        eff_medsam2_device=args.eff_medsam2_device,
        medsam2_device=args.medsam2_device,
        nninteractive_device=args.nninteractive_device,
        eff_medsam2_checkpoint=args.eff_medsam2_checkpoint,
        medsam2_checkpoint=args.medsam2_checkpoint,
        nninteractive_model_dir=args.nninteractive_model_dir,
        nninteractive_model_name=args.nninteractive_model_name,
        nninteractive_fold=args.nninteractive_fold,
        nninteractive_checkpoint=args.nninteractive_checkpoint,
        nninteractive_prompt=args.nninteractive_prompt,
        nninteractive_compile=args.nninteractive_compile,
        no_autozoom=args.no_autozoom,
        torch_threads=args.torch_threads,
        verbose=args.verbose,
        start_method=args.start_method,
    )
    print(json.dumps(results_to_dicts(results), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
