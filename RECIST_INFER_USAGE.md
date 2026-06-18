# recist_infer.py Usage

`recist_infer.py` is the pack-level entrypoint for RECIST-prompted lesion
segmentation. It gives us one command-line interface for model selection, image
loading, RECIST loading, model-native prompt conversion, and output writing.

The entrypoint is now intentionally RECIST-only: users provide an image and a
RECIST prompt. The script decides the best prompt conversion for each backend.
It does not expose direct point or direct box input.

## Environment

Run commands from wherever the pack has been placed:

```bash
cd /path/to/RECISTto3D
```

All built-in model paths are resolved relative to `recist_infer.py`, so moving
the whole `RECISTto3D` directory does not change the default checkpoint
locations.

Use the shared venv:

```bash
.venv/bin/python recist_infer.py --help
```

For venv setup and dependency validation, see `RECIST_PACK_VENV.md`.

## Default RECIST Conversion

The same RECIST input is converted differently per model family:

| Model | Default RECIST conversion | Why |
|---|---|---|
| `medsam2` | RECIST -> EAY canonical bbox | Best prompt in the EAY benchmark for MedSAM2. |
| `eff-medsam2` | RECIST -> EAY canonical bbox | Box is effectively tied with points for EfficientMedSAM2 and is the stable default. |
| `nninteractive` | RECIST -> EAY canonical 5 foreground points | Best RECIST-derived prompt in the EAY benchmark for nnInteractive. |

For MedSAM2/EfficientMedSAM2, the box is generated with the same canonical EAY
logic from `prompt_utils.py`: use the first and last rasterized RECIST pixels,
compute the RECIST diameter midpoint and length, then build a square
`[x_min,y_min,x_max,y_max]` box around that diameter.

For nnInteractive, the script samples 5 RECIST-line pixels with seed `2024` and
passes them as 3D foreground point interactions `(z,y,x)` on the RECIST slice.

## Minimal Commands

Run Efficient MedSAM2 with an input NPZ that already contains `imgs` and
`recist`:

```bash
.venv/bin/python recist_infer.py \
  --model eff-medsam2 \
  --image MedSAM2/data/validation_public_npz/CT_Lesion_FLARE23Ts_0007.npz \
  --output outputs/eff_small_pred.npz
```

Run Efficient MedSAM2 on a NIfTI image with a separate RECIST mask:

```bash
.venv/bin/python recist_infer.py \
  --model eff-medsam2 \
  --image image.nii.gz \
  --recist recist.nii.gz \
  --intensity window \
  --window=-175,275 \
  --output pred.npz \
  --output-nifti pred.nii.gz
```

Run with a RECIST line typed directly on the command line:

```bash
.venv/bin/python recist_infer.py \
  --model eff-medsam2 \
  --image image.nii.gz \
  --recist-line 64,108,229,160,281 \
  --intensity window \
  --window=-150,250 \
  --output pred.npz \
  --output-nifti pred.nii.gz
```

`--recist-line` format is:

```text
z,x1,y1,x2,y2
```

or, with an explicit label:

```text
z,x1,y1,x2,y2,label
```

It can be repeated for multiple lesions. When passing multiple RECIST lines,
each line must use a unique nonzero label:

```bash
.venv/bin/python recist_infer.py \
  --model eff-medsam2 \
  --image image.nii.gz \
  --recist-line 64,108,229,160,281,1 \
  --recist-line 72,210,180,255,220,2 \
  --intensity window \
  --window=-150,250 \
  --output pred.npz \
  --output-nifti pred.nii.gz
```

For NIfTI images, spacing is read from the image header. Only pass `--spacing`
when the image metadata is missing or wrong.

Run nnInteractive with the same RECIST mask:

```bash
.venv/bin/python recist_infer.py \
  --model nninteractive \
  --image image.nii.gz \
  --recist recist.nii.gz \
  --output pred_nninteractive.npz \
  --output-nifti pred_nninteractive.nii.gz
```

nnInteractive prompt choices:

- `--nninteractive-prompt 5_points` uses five foreground points sampled on the RECIST line. This is the default and matches the RECIST benchmark prompt.
- `--nninteractive-prompt 5pos_4neg` uses five foreground RECIST-line points plus four background points at RECIST-derived rotated-box corners.
- `--nninteractive-prompt 5pos_6neg` adds two more background points on the RECIST minor axis, outside the line center.

The positive/negative modes adapt the benchmark mask-derived `get_negative_pts`
geometry, but derive the negative points from the RECIST line endpoints only; no
GT mask is used.

Example positive/negative nnInteractive run:

```bash
.venv/bin/python recist_infer.py \
  --model nninteractive \
  --image image.nii.gz \
  --recist recist.nii.gz \
  --nninteractive-prompt 5pos_4neg \
  --output pred_nninteractive_posneg.npz \
  --output-nifti pred_nninteractive_posneg.nii.gz
```

By default nnInteractive uses the pack-local model directory:

```text
checkpoints/nnInteractive/nnInteractive_v1.0
```

You can still override it explicitly. Relative override paths are resolved from
the current working directory first, then from the pack root:

```bash
--nninteractive-model-dir checkpoints/nnInteractive/nnInteractive_v1.0
```

## Inputs

`--image` accepts:

- `.npz`: defaults to key `imgs`
- `.npy`
- `.nii` or `.nii.gz`
- 2D image files such as PNG/JPG

For NPZ images, override the image key with:

```bash
--image-key image
```

Spacing is stored internally as `z,y,x`. For NIfTI input, the script reads
SimpleITK spacing and converts from `x,y,z` to `z,y,x`.

For non-NIfTI inputs, or when metadata is missing, override spacing:

```bash
--spacing 5.0,0.75,0.75
```

## RECIST Inputs

You can provide RECIST in three ways:

1. `--recist recist.nii.gz`
2. one or more `--recist-line z,x1,y1,x2,y2[,label]`
3. an input NPZ that contains a `recist` key

`--recist` accepts NPZ/NPY/NIfTI/PNG. For NIfTI RECIST masks, the default
`--recist-space strict` requires the RECIST mask and image to have matching
size/spacing/origin/direction before inference starts. Use `--recist-space index`
only when the mask header is wrong but the array is already aligned to the image
index grid.

The RECIST mask used for inference must have the same `D,H,W` shape as the
loaded image after 2D inputs are promoted to one-slice volumes.

For a multi-lesion RECIST mask, each lesion should have a nonzero integer label.
The inference code processes each label separately. Each label must have RECIST
pixels on exactly one z slice.

No direct point or direct box input is supported. Those are derived internally
from RECIST according to the selected model.

## Model Choices

`--model eff-medsam2`

- Defaults to CPU unless `--device` is provided.
- Uses the Efficient MedSAM2 small checkpoint.
- Uses the pack-local `MedSAM2/checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt` checkpoint.
- Converts image intensity to uint8 `[0,255]`.
- Resizes frames to 512 when needed.
- Converts grayscale to 3-channel RGB.
- Applies ImageNet normalization.
- Converts RECIST to the EAY canonical bbox.
- Runs EfficientTAM video propagation.

`--model medsam2`

- Defaults to CUDA unless `--device` is provided.
- Uses the MedSAM2 RECIST baseline checkpoint and SAM2 video predictor.
- Uses the pack-local `MedSAM2/checkpoints/medsam2_FLARE25_RECIST_baseline.pt` checkpoint.
- Uses the same MedSAM2-style uint8/RGB/ImageNet preprocessing path.
- Converts RECIST to the EAY canonical bbox.

`--model nninteractive`

- Defaults to `cuda:0` unless `--device` is provided.
- Uses `checkpoints/nnInteractive/nnInteractive_v1.0` by default.
- Sends the raw image volume to `session.set_image(raw_image[None])`.
- Does not use MedSAM2's uint8/RGB/ImageNet preprocessing.
- Converts RECIST to 5 EAY canonical foreground points.
- Leaves crop/normalization behavior to nnInteractive's internal pipeline.

This separation is intentional. The shared entrypoint does not force both
frameworks through one preprocessing pipeline.

## Intensity Options

`--intensity` is MedSAM2/Efficient MedSAM2 only.

```bash
--intensity preserve
```

Use this when the input is already uint8 `[0,255]`. It raises an error if the
image has values outside that range.

```bash
--intensity minmax
```

Scale the whole image volume from its min/max range into uint8 `[0,255]`.

```bash
--intensity window --window=-175,275
```

Clip to the given CT window and scale to uint8 `[0,255]`. This is the usual
choice for raw CT NIfTI inputs in the EAY demo.

nnInteractive always receives the raw image, regardless of `--intensity`.

## Outputs

`--output pred.npz` writes a compressed NPZ with:

- `segs`: predicted segmentation as `D,H,W`
- `recist`: RECIST mask actually used by inference
- `boxes`: RECIST-derived boxes as `x1,y1,z1,x2,y2,z2`; this is populated for
  MedSAM2/EfficientMedSAM2 and empty for nnInteractive's 5-point default
- `spacing`: spacing as `z,y,x`
- `metadata`: JSON string with model, checkpoint, preprocessing, labels,
  source paths, prompt conversion, and runtime

`--output-nifti pred.nii.gz` optionally writes a NIfTI segmentation. For NIfTI
inputs, the output copies the input image metadata.

## EAY Demo

The hardcoded EAY demo uses this entrypoint once per cancer type:

```bash
.venv/bin/python run_eay_cancer_recist_demo.py
```

It copies one image and GT per cancer type into the pack root, generates
EAY-style RECIST masks from GT, runs `recist_infer.py`, and renders PNG panels.

Generated files include:

- `eay_demo_all_cancers.png`
- `eay_demo_selected_cases.json`
- `eay_demo_<cancer>_image.nii.gz`
- `eay_demo_<cancer>_gt.nii.gz`
- `eay_demo_<cancer>_recist.nii.gz`
- `eay_demo_<cancer>_pred.npz`
- `eay_demo_<cancer>_pred.nii.gz`
- `eay_demo_<cancer>_render.png`

## Quick Troubleshooting

If MedSAM2 complains about intensity range, use:

```bash
--intensity window --window=MIN,MAX
```

or:

```bash
--intensity minmax
```

If a RECIST label errors with "must have RECIST pixels on exactly one z slice",
check that each label in the RECIST mask is drawn on one slice only.

If nnInteractive errors that it cannot sample 5 points, check that the RECIST
line for each label has at least 5 pixels.

If nnInteractive cannot find its checkpoint, check that this pack-local file
exists:

```text
checkpoints/nnInteractive/nnInteractive_v1.0/fold_0/checkpoint_final.pth
```

Or pass another model folder explicitly:

```bash
--nninteractive-model-dir /path/to/nnInteractive_v1.0
```

If CUDA is unavailable, Efficient MedSAM2 can run on CPU by default:

```bash
--model eff-medsam2
```
