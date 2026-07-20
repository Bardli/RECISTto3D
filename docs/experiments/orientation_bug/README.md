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

The fix is scoped to the **click → model** (send) path only. Read NiiVue's
`vol.permRAS` and convert the RAS voxel from `frac2vox` back to the native stored
index:

- `pointFromEvent`: RAS voxel → native (`rasVoxToNative`) → then the existing
  fixed identity flips (via `nativeMaxForAxis`). The coordinate sent to the model
  is now direction-independent.

The **redraw** path (`lineScreenPointsFromVox`) is left on the original
`vox2frac([x1,y1,z])` mapping. An initial attempt to remap it via `permRAS`
mis-drew the RECIST line even for the identity examples: `vox2frac`/`frac2canvas`
already carry the matching display flips, so the original mapping round-trips
correctly for identity and remapping it double-corrected. Reverse-engineering
`vox2frac`/`frac2canvas` well enough to also fix the non-identity redraw was not
reliable from the source alone, so that display refinement is deferred (the
mask itself — which is what the model produces — is unaffected).

`write_nifti` (`recist_infer.py:885`, `CopyInformation`) was already correct and
is unchanged. Model-mask overlays are auto-reoriented to RAS by NiiVue and share
the input geometry, so they need no change.

## Regression tests (all pass)

- `test_orientation_roundtrip.py` — models NiiVue's native↔RAS reorientation;
  proves the old send-path diverges by direction and the fix does not, keeping the
  identity example bit-identical.
- `check_js.mjs` — the actual ported JS send-path helpers: identity / flipped-XY /
  swapped-XY / flip-X all map one click to the same native index.

Run:

```bash
python3 docs/experiments/orientation_bug/test_orientation_roundtrip.py
node    docs/experiments/orientation_bug/check_js.mjs
```
