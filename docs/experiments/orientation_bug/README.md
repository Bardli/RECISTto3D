# RECISTto3D orientation bug — root cause & fix

**Symptom:** when the uploaded volume's orientation (NIfTI direction matrix)
differs from the bundled examples, Gradio draws the predicted mask in the wrong
direction (mirrored / transposed).

## Root cause

Two coordinate spaces handle the direction matrix in opposite ways:

- **NiiVue viewer** (`app_three_models.py`, `pointFromEvent`): `nv.frac2vox`
  returns voxel indices in NiiVue's **RAS-reoriented** space. NiiVue 0.68.2's
  `convertFrac2Vox` source is literally commented `// dims === RAS`, and
  `calculateRAS` derives an axis permutation + per-axis flip (`permRAS`) from the
  affine.
- **Python model** (`recist_infer.py:143`): `sitk.GetArrayFromImage` returns the
  **native stored** buffer `(z, y, x)`, unchanged by the direction matrix
  (verified: changing direction leaves the buffer `np.array_equal`). RECIST
  `x1,y1` indexes `array[z, y1, x1]`.

The old line 485 (`vox: [isRadiological ? maxX - x : x, maxY - y, z]`) hardcoded
flips calibrated for **identity-direction** volumes only — which is all three
bundled examples (`diag(1,1,1)`, determinant 1). For any other direction NiiVue
applies an extra perm/flip that the hardcoded flips did not undo, so the model
received a mirrored/transposed coordinate.

## Fix (JS side, `app_three_models.py`)

Read NiiVue's `vol.permRAS` and convert between RAS and native voxel space:

- `pointFromEvent`: RAS voxel → native (`rasVoxToNative`) → then the existing
  fixed identity flips (via `nativeMaxForAxis`). Coordinate sent to the model is
  now direction-independent.
- `lineScreenPointsFromVox`: added `modelVoxToRasVox` (undo identity flip →
  `nativeToRasVox`) so a drawn line redraws on itself for any orientation.

`write_nifti` (`recist_infer.py:885`, `CopyInformation`) was already correct and
is unchanged. Model-mask overlays are auto-reoriented to RAS by NiiVue and share
the input geometry, so they need no change.

## Regression tests (all pass)

- `test_orientation_roundtrip.py` — models NiiVue's native↔RAS reorientation;
  proves the old code diverges by direction and the fix does not, keeping the
  identity example bit-identical.
- `check_js.mjs` — the actual ported JS helpers: identity / flipped-XY /
  swapped-XY / flip-X all map one click to the same native index.
- `check_roundtrip.mjs` — forward (`pointFromEvent`) and reverse
  (`modelVoxToRasVox`) are exact inverses over all `permRAS` variants.

Run:

```bash
python3 docs/experiments/orientation_bug/test_orientation_roundtrip.py
node    docs/experiments/orientation_bug/check_js.mjs
node    docs/experiments/orientation_bug/check_roundtrip.mjs
```
