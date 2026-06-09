#!/usr/bin/env python3
"""Run one RECIST-prompted segmentation demo per EAY cancer type.

By default this uses the pack-local copies under ``examples/``. Use
``--refresh-from-eay --eay-root /path/to/EAY131_50subset_NIFTI_v6_repatched``
to regenerate those copies from an external EAY dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import pdist, squareform


ROOT = Path(__file__).resolve().parent
RECIST_INFER = ROOT / "recist_infer.py"
EXAMPLES_DIR = ROOT / "examples"

SELECTED_CASES = {
    "colon_cancer": {
        "image": "colon_cancer/imagesTr/EAY131-1629428_acq1_1.3.6.1.4.1.14519.5.2.1.1620.1226.287796481956233263649228277433_0000.nii.gz",
        "gt": "colon_cancer/labelsTr/EAY131-1629428_acq1_1.3.6.1.4.1.14519.5.2.1.1620.1226.287796481956233263649228277433.nii.gz",
    },
    "kidney_cancer": {
        "image": "kidney_cancer/imagesTr/EAY131-1063620_acq1_1.3.6.1.4.1.14519.5.2.1.1620.1226.137777833365511372725105180306_0000.nii.gz",
        "gt": "kidney_cancer/labelsTr/EAY131-1063620_acq1_1.3.6.1.4.1.14519.5.2.1.1620.1226.137777833365511372725105180306.nii.gz",
    },
    "liver_cancer": {
        "image": "liver_cancer/imagesTr/EAY131-137353_acq1_1.3.6.1.4.1.14519.5.2.1.1620.1226.404574400461878346320568857921_0000.nii.gz",
        "gt": "liver_cancer/labelsTr/EAY131-137353_acq1_1.3.6.1.4.1.14519.5.2.1.1620.1226.404574400461878346320568857921.nii.gz",
    },
    "lung_cancer": {
        "image": "lung_cancer/imagesTr/EAY131-1318912_acq5_1.3.6.1.4.1.14519.5.2.1.1620.1226.318303435968648551176369216122_0000.nii.gz",
        "gt": "lung_cancer/labelsTr/EAY131-1318912_acq5_1.3.6.1.4.1.14519.5.2.1.1620.1226.318303435968648551176369216122.nii.gz",
    },
    "pancreas_cancer": {
        "image": "pancreas_cancer/imagesTr/EAY131-10017_acq2_1.3.6.1.4.1.14519.5.2.1.1620.1226.140098505584939297262476787866_0000.nii.gz",
        "gt": "pancreas_cancer/labelsTr/EAY131-10017_acq2_1.3.6.1.4.1.14519.5.2.1.1620.1226.140098505584939297262476787866.nii.gz",
    },
}


@dataclass
class DemoResult:
    cancer: str
    image_copy: Path
    gt_copy: Path
    recist_path: Path
    pred_npz: Path
    pred_nifti: Path
    render_png: Path
    z: int
    recist_line: tuple[int, int, int, int, int]
    dice_3d: float


def read_nifti(path: Path):
    img = sitk.ReadImage(str(path))
    return img, sitk.GetArrayFromImage(img)


def write_like(path: Path, array: np.ndarray, reference: sitk.Image) -> None:
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path))


def resolve_pack_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_eay_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def copy_case_files(cancer: str, image_path: Path, gt_path: Path, examples_dir: Path) -> tuple[Path, Path]:
    examples_dir.mkdir(parents=True, exist_ok=True)
    image_copy = examples_dir / f"eay_demo_{cancer}_image.nii.gz"
    gt_copy = examples_dir / f"eay_demo_{cancer}_gt.nii.gz"
    shutil.copy2(image_path, image_copy)
    shutil.copy2(gt_path, gt_copy)
    return image_copy, gt_copy


def get_case_files(cancer: str, spec: dict[str, str], args, examples_dir: Path) -> tuple[Path, Path]:
    image_copy = examples_dir / f"eay_demo_{cancer}_image.nii.gz"
    gt_copy = examples_dir / f"eay_demo_{cancer}_gt.nii.gz"
    if image_copy.exists() and gt_copy.exists() and not args.refresh_from_eay:
        return image_copy, gt_copy

    if args.eay_root is None:
        raise FileNotFoundError(
            f"Missing pack-local demo files for {cancer}: {image_copy}, {gt_copy}. "
            "Pass --refresh-from-eay --eay-root /path/to/EAY131_50subset_NIFTI_v6_repatched "
            "to recreate them from the external EAY dataset."
        )

    eay_root = resolve_eay_path(args.eay_root)
    image_path = eay_root / spec["image"]
    gt_path = eay_root / spec["gt"]
    if not image_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"EAY case files not found: {image_path}, {gt_path}")
    return copy_case_files(cancer, image_path, gt_path, examples_dir)


MIN_LESION_VOXELS = 10
RECIST_LINE_THICKNESS = 2


def connected_components(mask: np.ndarray) -> np.ndarray:
    try:
        import cc3d

        return cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    except ImportError:
        from scipy import ndimage

        structure = np.ones((3, 3, 3), dtype=np.uint8)
        instance, _ = ndimage.label(mask.astype(np.uint8), structure=structure)
        return instance


def compute_recist_line(mask_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import cv2

        contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        pts = np.vstack(contours).squeeze()
    except ImportError:
        from skimage import measure

        contours_yx = measure.find_contours(mask_2d.astype(np.uint8), 0.5)
        if not contours_yx:
            return None
        pts_yx = np.concatenate(contours_yx, axis=0)
        pts = np.stack([pts_yx[:, 1], pts_yx[:, 0]], axis=1)

    if pts.ndim != 2 or len(pts) < 2:
        return None
    if len(pts) > 500:
        pts = pts[np.linspace(0, len(pts) - 1, 500, dtype=int)]

    dist_matrix = squareform(pdist(pts))
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    return np.rint(pts[max_idx[0]]).astype(int), np.rint(pts[max_idx[1]]).astype(int)


def draw_thick_line(mask: np.ndarray, z: int, x1: int, y1: int, x2: int, y2: int, label: int) -> None:
    try:
        import cv2

        cv2.line(mask[z], (int(x1), int(y1)), (int(x2), int(y2)), color=int(label), thickness=RECIST_LINE_THICKNESS)
        return
    except ImportError:
        pass

    steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = np.rint(np.linspace(x1, x2, steps)).astype(int)
    ys = np.rint(np.linspace(y1, y2, steps)).astype(int)
    radius = max(0, RECIST_LINE_THICKNESS // 2)
    for x, y in zip(xs, ys):
        y0, y1_clip = max(0, y - radius), min(mask.shape[1], y + radius + 1)
        x0, x1_clip = max(0, x - radius), min(mask.shape[2], x + radius + 1)
        mask[z, y0:y1_clip, x0:x1_clip] = label


def generate_recist_from_gt(gt: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int, int]]:
    binary = gt > 0
    if not np.any(binary):
        raise ValueError("GT is empty; cannot generate RECIST")

    instance = connected_components(binary)
    recist = np.zeros_like(instance, dtype=np.uint16)
    focus_line = None
    focus_size = -1

    lesion_ids = [int(x) for x in np.unique(instance) if x != 0]
    for lid in lesion_ids:
        lesion_mask = (instance == lid).astype(np.uint8)
        lesion_size = int(lesion_mask.sum())
        if lesion_size < MIN_LESION_VOXELS:
            continue

        area_per_slice = np.sum(lesion_mask, axis=(1, 2))
        z = int(np.argmax(area_per_slice))
        result = compute_recist_line(lesion_mask[z])
        if result is None:
            continue

        p1, p2 = result
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        draw_thick_line(recist, z, x1, y1, x2, y2, lid)
        if lesion_size > focus_size:
            focus_size = lesion_size
            focus_line = (z, x1, y1, x2, y2)

    if focus_line is None:
        raise ValueError("No GT connected component was large enough to generate RECIST")
    return recist, focus_line


def run_inference(args, image_copy: Path, recist_path: Path, pred_npz: Path, pred_nifti: Path) -> None:
    cmd = [
        sys.executable,
        str(RECIST_INFER),
        "--model",
        args.model,
        "--image",
        str(image_copy),
        "--recist",
        str(recist_path),
        "--intensity",
        args.intensity,
        "--output",
        str(pred_npz),
        "--output-nifti",
        str(pred_nifti),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.intensity == "window":
        cmd.append(f"--window={args.window}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_mask = pred > 0
    gt_mask = gt > 0
    denom = int(pred_mask.sum() + gt_mask.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred_mask, gt_mask).sum() / denom)


def window_image(image_slice: np.ndarray, window: str) -> np.ndarray:
    lo, hi = [float(v) for v in window.replace(",", " ").split()]
    out = np.clip(image_slice.astype(np.float32), lo, hi)
    return (out - lo) / (hi - lo)


def add_mask(ax, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba)


def overlay(ax, base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    add_mask(ax, mask, color, alpha)


def render_case(
    cancer: str,
    image: np.ndarray,
    gt: np.ndarray,
    recist: np.ndarray,
    pred: np.ndarray,
    z: int,
    dice_3d: float,
    out_path: Path,
    window: str,
) -> None:
    base = window_image(image[z], window)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fig.suptitle(f"{cancer} | z={z} | 3D Dice={dice_3d:.3f}", fontsize=12)

    overlay(axes[0], base, recist[z] > 0, (1.0, 0.0, 0.0), 0.95)
    axes[0].set_title("RECIST prompt")

    overlay(axes[1], base, gt[z] > 0, (0.0, 0.9, 0.1), 0.45)
    axes[1].set_title("GT")

    overlay(axes[2], base, pred[z] > 0, (0.1, 0.35, 1.0), 0.45)
    axes[2].set_title("Prediction")

    axes[3].imshow(base, cmap="gray", vmin=0, vmax=1)
    add_mask(axes[3], gt[z] > 0, (0.0, 0.9, 0.1), 0.35)
    add_mask(axes[3], pred[z] > 0, (0.1, 0.35, 1.0), 0.35)
    axes[3].set_title("GT + Prediction")

    for ax in axes:
        ax.axis("off")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_summary(results: list[DemoResult], window: str, examples_dir: Path) -> None:
    rows = len(results)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]

    for r, result in enumerate(results):
        _, image = read_nifti(result.image_copy)
        _, gt = read_nifti(result.gt_copy)
        _, recist = read_nifti(result.recist_path)
        pred = np.load(result.pred_npz)["segs"]
        z = result.z
        base = window_image(image[z], window)
        panels = [
            (recist[z] > 0, (1.0, 0.0, 0.0), "RECIST"),
            (gt[z] > 0, (0.0, 0.9, 0.1), "GT"),
            (pred[z] > 0, (0.1, 0.35, 1.0), "Prediction"),
            (gt[z] > 0, (0.0, 0.9, 0.1), "GT + Prediction"),
        ]
        for c, (mask, color, title) in enumerate(panels):
            ax = axes[r, c]
            if title == "GT + Prediction":
                ax.imshow(base, cmap="gray", vmin=0, vmax=1)
                add_mask(ax, gt[z] > 0, (0.0, 0.9, 0.1), 0.35)
                add_mask(ax, pred[z] > 0, (0.1, 0.35, 1.0), 0.35)
            else:
                overlay(ax, base, mask, color, 0.45 if title != "RECIST" else 0.95)
            ax.set_title(f"{result.cancer} | {title} | Dice {result.dice_3d:.3f}", fontsize=9)
            ax.axis("off")

    fig.savefig(examples_dir / "eay_demo_all_cancers.png", dpi=180)
    plt.close(fig)


def run_case(cancer: str, spec: dict[str, str], args, examples_dir: Path) -> DemoResult:
    image_copy, gt_copy = get_case_files(cancer, spec, args, examples_dir)

    image_ref, image = read_nifti(image_copy)
    _, gt = read_nifti(gt_copy)
    recist, recist_line = generate_recist_from_gt(gt)

    recist_path = examples_dir / f"eay_demo_{cancer}_recist.nii.gz"
    pred_npz = examples_dir / f"eay_demo_{cancer}_pred.npz"
    pred_nifti = examples_dir / f"eay_demo_{cancer}_pred.nii.gz"
    render_png = examples_dir / f"eay_demo_{cancer}_render.png"

    write_like(recist_path, recist, image_ref)
    run_inference(args, image_copy, recist_path, pred_npz, pred_nifti)

    pred = np.load(pred_npz)["segs"]
    dice_3d = dice_score(pred, gt)
    render_case(cancer, image, gt, recist, pred, recist_line[0], dice_3d, render_png, args.window)

    return DemoResult(
        cancer=cancer,
        image_copy=image_copy,
        gt_copy=gt_copy,
        recist_path=recist_path,
        pred_npz=pred_npz,
        pred_nifti=pred_nifti,
        render_png=render_png,
        z=recist_line[0],
        recist_line=recist_line,
        dice_3d=dice_3d,
    )


def pack_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_manifest(results: list[DemoResult], examples_dir: Path) -> None:
    payload = []
    for result in results:
        payload.append(
            {
                "cancer": result.cancer,
                "image_copy": pack_relative(result.image_copy),
                "gt_copy": pack_relative(result.gt_copy),
                "recist": pack_relative(result.recist_path),
                "prediction_npz": pack_relative(result.pred_npz),
                "prediction_nifti": pack_relative(result.pred_nifti),
                "render_png": pack_relative(result.render_png),
                "z": result.z,
                "recist_line_zxyxy": list(result.recist_line),
                "dice_3d": result.dice_3d,
            }
        )
    (examples_dir / "eay_demo_selected_cases.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("eff-medsam2", "medsam2"), default="eff-medsam2")
    parser.add_argument("--device", help="Torch device passed to recist_infer.py.")
    parser.add_argument("--intensity", choices=("window", "minmax", "preserve"), default="window")
    parser.add_argument("--window", default="-175,275", help="CT display/model window for MedSAM2-style inputs.")
    parser.add_argument("--examples-dir", default="examples", help="Pack-relative output/input directory for demo copies.")
    parser.add_argument("--eay-root", help="External EAY131_50subset_NIFTI_v6_repatched root used with --refresh-from-eay.")
    parser.add_argument("--refresh-from-eay", action="store_true", help="Refresh pack-local demo inputs from --eay-root.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    examples_dir = resolve_pack_path(args.examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cancer, spec in SELECTED_CASES.items():
        print(f"=== {cancer} ===", flush=True)
        result = run_case(cancer, spec, args, examples_dir)
        print(f"wrote {result.render_png} | Dice={result.dice_3d:.3f}", flush=True)
        results.append(result)
    make_summary(results, args.window, examples_dir)
    write_manifest(results, examples_dir)
    print(f"wrote {examples_dir / 'eay_demo_all_cancers.png'}")
    print(f"wrote {examples_dir / 'eay_demo_selected_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
