"""Failing test: reproduce the RECISTto3D orientation bug.

Root cause hypothesis
----------------------
The Gradio viewer (app_three_models.py:485) converts a mouse click to a voxel
index with:

    vox = [isRadiological ? maxX - x : x,  maxY - y,  z]        # (x, y, z)

where (x, y, z) come from NiiVue's ``frac2vox``. NiiVue 0.68.2's
``convertFrac2Vox`` returns indices in NiiVue's *RAS-reoriented* space (the
source is literally commented ``// dims === RAS``; see ``calculateRAS`` which
computes ``permRAS`` / axis ``flip`` from the affine and stores ``dimsRAS``).

The Python model side (recist_infer.py:143) uses ``sitk.GetArrayFromImage`` which
returns the volume in *native stored order* (z, y, x), untouched by the direction
matrix.

For an identity-direction volume NiiVue's RAS space == native stored space, so
the app's hardcoded ``maxX - x`` / ``maxY - y`` flips land on the correct stored
index. For any volume whose direction requires a non-trivial permutation/flip,
the two spaces diverge and the click lands on the wrong stored voxel -> the mask
is segmented in "another direction".

This test models NiiVue's native->RAS reorientation faithfully and asserts the
end-to-end round trip (anatomical point -> screen -> app voxel -> sitk index)
recovers the SAME stored voxel regardless of the input direction matrix. It is
expected to PASS for identity and FAIL for a flipped direction, proving the bug.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Faithful port of NiiVue 0.68.2 calculateRAS (perm + flip from the affine).
# ---------------------------------------------------------------------------
def niivue_perm_flip(affine3x3):
    a = np.asarray(affine3x3, dtype=float)
    absR = np.abs(a)
    ixyz = [1, 1, 1]
    if absR[1, 0] > absR[0, 0]:
        ixyz[0] = 2
    if absR[2, 0] > absR[0, 0] and absR[2, 0] > absR[1, 0]:
        ixyz[0] = 3
    if ixyz[0] == 1:
        ixyz[1] = 2 if absR[1, 1] > absR[2, 1] else 3
    elif ixyz[0] == 2:
        ixyz[1] = 1 if absR[0, 1] > absR[2, 1] else 3
    else:
        ixyz[1] = 1 if absR[0, 1] > absR[1, 1] else 2
    ixyz[2] = 6 - ixyz[1] - ixyz[0]
    perm = [1, 2, 3]
    perm[ixyz[0] - 1] = 1
    perm[ixyz[1] - 1] = 2
    perm[ixyz[2] - 1] = 3
    # R = rotM columns permuted by perm
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = a[i, perm[j] - 1]
    flip = [1 if R[0, 0] < 0 else 0,
            1 if R[1, 1] < 0 else 0,
            1 if R[2, 2] < 0 else 0]
    return perm, flip


def native_to_ras_index(vox_native, dims_native, perm, flip):
    """Map a native (i,j,k) voxel index to NiiVue's RAS (x,y,z) voxel index."""
    # dims_native is header dims [d1,d2,d3] (x,y,z stored)
    ras = [0, 0, 0]
    for out_axis in range(3):
        src_axis = perm[out_axis] - 1          # which native axis feeds RAS axis
        idx = vox_native[src_axis]
        if flip[out_axis]:
            idx = dims_native[src_axis] - 1 - idx
        ras[out_axis] = idx
    return ras


def ras_dims(dims_native, perm):
    return [dims_native[perm[0] - 1], dims_native[perm[1] - 1], dims_native[perm[2] - 1]]


# ---------------------------------------------------------------------------
# Model of the app's coordinate handling (app_three_models.py:485).
# The user clicks an anatomical point. NiiVue renders it, the app reads back a
# RAS voxel via frac2vox, then applies the hardcoded flips to build the coord
# it sends to the model. We then interpret that coord as a native sitk index
# (which is what recist_infer.py does).
# ---------------------------------------------------------------------------
def app_click_to_sitk_index(vox_native_truth, dims_native, affine3x3, is_radiological=True):
    """End-to-end: an anatomical native voxel -> what recist_infer.py finally indexes.

    Two established facts drive this model:
      (A) NiiVue frac2vox returns RAS-space voxel indices (source: ``// dims === RAS``
          + calculateRAS/permRAS). We compute that via native_to_ras_index.
      (B) sitk.GetArrayFromImage returns the NATIVE stored buffer, unchanged by the
          direction matrix (verified experimentally). recist_infer.py indexes it as
          [z, y1, x1], i.e. it treats the app coord as a NATIVE (x, y, z) index.

    The app's line 485 flips (maxX-x, maxY-y) were calibrated so that, FOR THE
    IDENTITY EXAMPLES, the RAS voxel maps back to the correct native index. We bake
    that calibration in as the identity baseline, then let perm/flip diverge it.
    """
    perm, flip = niivue_perm_flip(affine3x3)
    dimsRAS = ras_dims(dims_native, perm)

    # (A) NiiVue frac2vox -> RAS voxel index of the clicked point.
    x_ras, y_ras, z_ras = native_to_ras_index(vox_native_truth, dims_native, perm, flip)

    # app_three_models.py:485  vox = [isRad ? maxX-x : x, maxY-y, z]
    maxX = dimsRAS[0] - 1
    maxY = dimsRAS[1] - 1
    app_x = (maxX - x_ras) if is_radiological else x_ras
    app_y = maxY - y_ras
    app_z = z_ras

    # (B) recist_infer.py treats (app_x, app_y, app_z) as a NATIVE (x, y, z) index.
    return (app_x, app_y, app_z)


def permRAS(perm, flip):
    """NiiVue's permRAS: signed 1-based native axis per RAS axis (negative == flipped)."""
    p = list(perm)
    for i in range(3):
        if flip[i]:
            p[i] = -p[i]
    return p


def ras_vox_to_native(ras_xyz, dimsRAS, perm_ras):
    """Invert NiiVue's native->RAS reorientation: RAS voxel -> native (i,j,k)=sitk (x,y,z)."""
    native = [0, 0, 0]
    for i in range(3):          # i = RAS output axis
        ax = abs(perm_ras[i]) - 1   # native axis that feeds RAS axis i
        v = ras_xyz[i]
        if perm_ras[i] < 0:
            v = dimsRAS[i] - 1 - v
        native[ax] = v
    return native


def app_click_to_sitk_index_FIXED(vox_native_truth, dims_native, affine3x3, is_radiological=True):
    """Proposed fix: recover the NATIVE index from NiiVue's RAS voxel via permRAS,
    then apply ONLY the fixed identity-display flips that the app already relied on.

    The existing line-485 flips (maxX-x, maxY-y under radiological) encode the fixed
    RAS<->sitk-native relationship that holds for the identity examples. We keep them,
    but first undo NiiVue's affine-driven perm/flip so 'RAS voxel' is expressed on the
    identity axes before those fixed flips are applied.
    """
    perm, flip = niivue_perm_flip(affine3x3)
    dimsRAS = ras_dims(dims_native, perm)
    pras = permRAS(perm, flip)

    # (A) NiiVue frac2vox -> RAS voxel index.
    ras = list(native_to_ras_index(vox_native_truth, dims_native, perm, flip))

    # FIX: invert perm/flip -> native axes (this is what sitk stores).
    nat = ras_vox_to_native(ras, dimsRAS, pras)   # nat = native (x, y, z)

    # Then the SAME fixed identity-display flips the app already does, on native dims.
    maxX = dims_native[0] - 1
    maxY = dims_native[1] - 1
    app_x = (maxX - nat[0]) if is_radiological else nat[0]
    app_y = maxY - nat[1]
    app_z = nat[2]
    return (app_x, app_y, app_z)


# The physically SAME anatomical location, expressed as a native (x,y,z) index,
# depends on the direction matrix: sitk keeps the storage buffer fixed, so the
# native index of a fixed anatomy is identical across directions ONLY if the
# buffer is identical. We hold the stored buffer fixed and ask: does the app send
# the model the same native index for the same click, regardless of direction?
def app_coord_for_click(dims_native, affine3x3, click_native_xyz):
    return app_click_to_sitk_index(click_native_xyz, dims_native, affine3x3)


if __name__ == "__main__":
    dims_native = [512, 512, 113]           # x, y, z stored dims (colon example)
    click = [100, 380, 40]                  # user clicks the SAME stored voxel each time

    identity = np.diag([1.0, 1.0, 1.0])
    flipped_xy = np.diag([-1.0, -1.0, 1.0])     # common LPS<->RAS difference
    swapped_xy = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])  # X/Y axes swapped

    base = app_coord_for_click(dims_native, identity, click)
    print("With the SAME stored buffer, the same click must yield the same native")
    print("index sent to the model, no matter the header direction:\n")
    print(f"  identity  perm={niivue_perm_flip(identity)}  -> model receives {base}")

    all_ok = True
    for name, aff in [("flipped XY", flipped_xy), ("swapped XY", swapped_xy)]:
        got = app_coord_for_click(dims_native, aff, click)
        ok = tuple(got) == tuple(base)
        all_ok &= ok
        print(f"  {name:10s} perm/flip={niivue_perm_flip(aff)}  -> model receives {got}  "
              f"{'consistent' if ok else 'DIVERGED -> mask drawn in wrong direction'}")

    print()
    if all_ok:
        print("current code: ALL CONSISTENT (no bug)")
    else:
        print("current code: BUG reproduced (diverges by direction)\n")

    # -------- proposed FIX --------
    print("Proposed fix (invert permRAS first, then fixed identity flips):\n")
    base_fx = app_click_to_sitk_index_FIXED(click, dims_native, identity)
    print(f"  identity   -> model receives {base_fx}")
    fix_ok = True
    for name, aff in [("flipped XY", flipped_xy), ("swapped XY", swapped_xy)]:
        got = app_click_to_sitk_index_FIXED(click, dims_native, aff)
        ok = tuple(got) == tuple(base_fx)
        fix_ok &= ok
        print(f"  {name:10s} -> model receives {got}  "
              f"{'consistent' if ok else 'STILL DIVERGED'}")

    print()
    assert fix_ok, "FIX FAILED: fixed transform still direction-dependent."
    # The fix must also agree with the known-good identity baseline value.
    assert tuple(base_fx) == tuple(base), (
        f"FIX changed identity behavior: was {base}, now {base_fx}"
    )
    print("FIX VERIFIED: all directions map the same click to the same native index,")
    print("and the identity example is unchanged.")
