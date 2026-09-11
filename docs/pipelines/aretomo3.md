# AreTomo3: projection, motion and CETS

For worked commands, see [native → CETS](../tutorials/aretomo3-to-cets.md)
and [CETS → native](../tutorials/cets-to-aretomo3.md).

This document describes the AreTomo3 models implemented by `cets-nonrigid`:
`.aln` tilt-series alignment and `.mcaln` frame motion. They use different local
fields and shift signs. The package names are `cets-nonrigid` and
`cets_nonrigid`; the CLI format names are `aretomo3` and `mcaln`.

## Tilt-series coordinates and projection

An `.aln` alignment uses pixels of the motion-corrected, unbinned tilt images.
Reconstruction binning (`-AtBin`) does not rescale those alignment values.
Projection maps a 3D reference-volume point to its sampling position in the raw
motion-corrected tilt image. An aligned-stack display transformation, including
AreTomo's possible XY transposition, is not part of this map.

For the following equations, `s` is image pixel size in Å, `V` is reference-volume
extent in Å, and `N` is raw image size in pixels. The native wrapper takes
corner-origin volume coordinates `P` in Å and converts them to AreTomo's fit frame:

```text
(X, Y, Z) = (P - V/2) / s
(Cx, Cy, Cz) = (X, Y, -Z)

fx = Cx*cos(TILT) - Cz*sin(TILT)
u0 = fx*cos(ROT) - Cy*sin(ROT)
v0 = fx*sin(ROT) + Cy*cos(ROT)

q_native(P) = s * ((u0, v0) + L_t(u0, v0) + (TX, TY) + N/2)
```

Angles are in degrees. `(u0,v0)` is the centered, rotated coordinate frame of the
`.aln` `CoordX/CoordY` columns. Local and global shifts are **added**. `L_t` is
sampled before the global shift is applied.

The local field uses each tilt's projected patch centers `c_j`, residual shifts
`l_j` and `Good` values:

```text
rho_j² = ((u0 - CoordX_j)/Nx)² + ((v0 - CoordY_j)/Ny)²
w_j = exp(-100*rho_j²) for Good_j >= 0.9; otherwise 0
L_t = sum_j(w_j*l_j) / sum_j(w_j)
```

There is no distance cutoff; no good patches gives zero local shift. Patch
coordinates change with tilt and must not be treated as one fixed image grid.
The default stable implementation rescales the exponential weights to avoid
underflow. Compatibility mode reproduces native float32 arithmetic, including
possible nonfinite results if all contributing weights underflow.

`TILT` already includes `AlphaOffset`. Applying that offset again rotates twice.
`BetaOffset` does not enter this projection model and cannot stand in for a fitted
X rotation; the adapter preserves it as provenance. The fit-frame Z inversion is
an internal convention conversion, not an extra flip to apply to CETS coordinates.

## Correspondence to CETS

CETS physical coordinates have origin at array index `floor(size/2)`, in a
right-handed frame measured in Å. Define `C_v = floor(volume_size/2)*volume_spacing`
and `C_i = floor(image_size/2)*image_spacing`. For CETS point `p`:

```text
p in CETS reference volume
    | + C_v
    v
P in native corner-origin volume
    | AreTomo full projection, including patches
    v
q_native in raw image Å
    | - C_i
    v
q in CETS tilt-image coordinates
    |
    +---- r = q - G_CETS(p) ----> residual payload
```

The shared rigid codec creates `Alignment.projection_alignments` from ROT, TILT
and TX/TY. With standard right-handed column-vector rotations, the corresponding
rotation is `Rz(ROT) @ Ry(TILT)`. Center differences and pixel-to-Å scaling belong
in the codec's coordinate conversion. In particular, `N/2` and `floor(N/2)` differ
by half a pixel for odd dimensions; do not assume those centers coincide.

| Native information | CETS representation |
|---|---|
| ROT, TILT, TX/TY | Global projection operators `G_t` |
| Local Coord/Shift/Good patch model | Sampled `projected_residual`, fitting masks and weights |
| Volume dimensions and spacing | Referenced `Tomogram`, whose image path may be null |
| Raw sections and dark tilts | `TiltImage` IDs, projection associations and alignment exclusions |
| Acquisition order, dose, stage angles | Image acquisition metadata and angle observations |
| CTF companion | Per-image CTF metadata; angles/phase normalized to degrees |
| AlphaOffset/BetaOffset and native files | Alignment provenance |

Native local rows follow the dark-removed, tilt-sorted list; SEC is the native
1-based association to raw sections. CETS payload rows contain every tilt image,
including dark rows, and use `tilt_image_ids` explicitly. Acquisition order is
separate. A row without a projection alignment has no G and no projected observation;
its residual is zero-filled and masked. Unknown stage angles are not invented from
an effective projection angle.

## Import and target fitting

`api.to_cets("aretomo3", ...)` loads the native model and uses the shared rigid
codec for G. Sampling evaluates native q and document G separately, checks the
rigid baseline, and stores `r = q - G` in float32. The `.aln` patch model provides
an image displacement; it does not supply a 3D trajectory or sampled CTF-depth
channel. Per-tilt CTF parameters remain available as core metadata.

On `api.fit(bundle, "aretomo3", ...)`, the package reconstructs q from CETS G+r,
fits representable AreTomo globals, then fits patch shifts against the target's
own pre-global-shift coordinates. Patch depth can use the validated least-squares
construction or zero. An unrepresentable X tilt fails the global tolerance gate;
local patches are not allowed to hide that global mismatch. The AreTomo3 project
writer prepares the corresponding stack, angle/CTF companions and `-Cmd 2` layout.

The exchange preserves observables at samples, not patch coefficients or a unique
continuous field. A target fit's held-out error measures the approximation.

## Movie motion (`.mcaln`)

Movie coordinates are corner-origin pixel indices at the alignment-image spacing.
For corrected-image position `x_px`, the motion model gives its source position in
raw frame `f`:

```text
rho_j = sqrt(((x_px.x-cx_j)/Nx)² + ((x_px.y-cy_j)/Ny)²)
w_j = exp(-100*rho_j), only for valid patches with rho_j <= 0.5
S_f(x_px) = weighted average of local patch corrections
q_raw = x_px - S_f(x_px) - global_shift_f
```

This kernel uses linear distance and a hard cutoff; tilt-series patches use
squared distance and no cutoff. Movie corrections are **subtracted**, unlike
`.aln` local shifts. The field is evaluated at the corrected/output coordinate.
The native correction-chain and CUDA kernel checks establish this composition.

In CETS, `MovieStack` owns the frame geometry and `MovieAlignment` owns a
`FrameAlignment` translation of `-s*global_shift_f`. The local residual is
`-s*S_f` at each sampled point. Both sides use the same image frame, so recentering
cancels from a pure translation. `frame_ids` identifies the payload rows.

The `.mcaln` frame table maps raw frames to integrated frames and retained aligned
frames. CETS preserves those integration spans, exclusions and the `fmRef` gauge;
`reference_frame_id` refers to an image identity, not a raw-file array offset.
The format's native binary import/export requires the AreTomo3 motion-I/O patch.

## Implementation and checks

- [Tilt projection and patch kernel](../../src/cets_nonrigid/models/aretomo_ts.py),
  [native frame conversions](../../src/cets_nonrigid/frames.py), and
  [CETS adapter](../../src/cets_nonrigid/native.py).
- [AreTomo target fitting](../../src/cets_nonrigid/fit/aretomo_ts_fit.py) and
  [motion model](../../src/cets_nonrigid/models/aretomo_motion.py).
- [Tilt-model checks](../../tests/test_aretomo_ts_model.py),
  [native CUDA contract](../../tests/test_m6_kernel_contract.py), and
  [CETS boundary checks](../../tests/test_cets_exchange.py).

See the [shared exchange profile](../exchange-profile.md) for payload masks,
precision, immutable attachment and digest rules, and the
[.mcaln format](../native-mcaln_format.md) for its native text layout.
