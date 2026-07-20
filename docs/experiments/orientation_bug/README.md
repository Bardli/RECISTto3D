# RECISTto3D orientation bug — root cause & fix

**Symptom:** for some volumes the drawn RECIST line jumps to the opposite corner
("diagonal") on mouse-up, and/or the predicted mask is drawn mirrored.

## Root cause (verified against a live NiiVue + NIfTI headers)

NiiVue displays a volume using the **NIfTI affine** (RAS+ convention). The model
side (`recist_infer.py`, `sitk.GetArrayFromImage`) reads the volume in native
stored order using the **SimpleITK direction** (LPS+ convention). NIfTI and ITK
negate the X and Y axis signs relative to each other, so a single file is read
with *opposite* orientation by the two:

| file | NIfTI affine diag (NiiVue) | sitk direction diag (model) |
|------|----------------------------|-----------------------------|
| `kidney_cancer.nii.gz`   | −0.68, −0.68, + | **+1, +1, +1** |
| `pancreas_cancer.nii.gz` | +0.94, +0.94, + | **−1, −1, +1** |

The app's RECIST coordinate flips were calibrated for the kidney-style layout
(sitk-identity). `pancreas_cancer.nii.gz` has the mirror storage layout, so the
click→coordinate and coordinate→redraw mappings disagree and the line/mask land
in the mirrored (diagonal) position.

Note this is **not** NiiVue's `permRAS` axis permutation — a live NiiVue reports
`permRAS = [1,2,3]` (identity) for pancreas. The divergence is purely the
RAS/LPS *sign* convention, which `permRAS` does not expose. Earlier `permRAS`-
based JS fixes were therefore inert for the real failing file and were reverted.

## Fix (Python side, `app_three_models.py`)

Normalise every volume to a single canonical orientation before it reaches
either the viewer or the model, so both agree:

- `_normalize_orientation(src, dst)` — `sitk.DICOMOrient(img, "LPS")` (skips the
  rewrite when the direction is already identity). Geometry is preserved (a fixed
  physical point keeps its intensity; verified).
- `_copy_upload` normalises uploads in place of a plain copy.
- `_example_normalized` serves a cached canonical copy of the read-only example
  files (never mutates the repo examples).

After normalisation `pancreas_cancer.nii.gz` matches the known-good kidney
convention exactly (sitk `(1,1,1)`, NIfTI affine `(−,−,+)`), so the viewer and
model are consistent and the original coordinate flips are valid again. The JS
coordinate code is back to its original pre-bug form.

## Probe / regression aids

- `nv_probe.mjs` / `run_probe.mjs` / `probe.html` — drive a real headless NiiVue
  0.68.2 to read `permRAS` / `frac2vox` for a volume. This is how `permRAS=[1,2,3]`
  for pancreas was established, ruling out the permutation hypothesis. Needs
  `puppeteer-core` + system Chrome; `frac2canvas` returns null headless (no
  on-screen layout), so screen-space assertions must be checked in the real app.
- `test_orientation_roundtrip.py` / `check_js.mjs` — earlier permRAS models,
  retained for history; they describe the rejected hypothesis, not the shipped
  fix.
