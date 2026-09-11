> Frozen source-kernel reference. Original arewarpion names and native corner frames
> are retained here for port auditing; the public exchange contract is in
> [exchange-profile.md](exchange-profile.md).

# arewarpion coordinate & convention contracts (tilt series: FROZEN at M0)

Status: tilt-series contracts below are **frozen**. Frame-series (motion) map
semantics are **provisional** until the M6 correction-kernel contract test
(see `mcaln_format.md`).

All constants live in `arewarpion.conventions`; nothing else may hard-code a
cross-tool sign.

## Canonical frame (= Warp's)

- **Image**: raw motion-corrected tilt image (or movie frame). Angstrom,
  corner origin, pixel `i` centered at `i * s` where `s` is the pixel size in
  A/px of that image.
- **Volume**: tomogram XYZ in Angstrom, corner origin (Warp
  `VolumeDimensionsAngstrom` frame). The volume center projects to the image
  center under zero shifts. Tilt axis is parallel to volume +Y in both tools'
  models.

Both tools' models map to the *raw* image, so AreTomo's aligned-stack XY
transposition (`|sin ROT| > 0.707`) never enters the conversion.

## Warp tilt-series forward chain (reference: `TiltSeries.cs:402`, warpylib `positions.py`)

```
p' = p - V/2 + VolumeWarp(x/Vx, y/Vy, z/Vz, dose_frac)     # quadrilinear LinearGrid4D
q  = Rz(+axisAngle_t) . Ry(-(angle_t + levelY)) . Rx(+levelX) . p'
q_xy += (axisOffsetX_t, axisOffsetY_t) + I/2
out = q_xy - GridMovement(qx/Ix, qy/Iy, t/(N-1))            # cubic, sampled at PRE-correction pos
```

- Angles in file order (typically descending after WarpTools import).
- `GridVolumeWarp*` uses normalized dose as 4th axis; every other grid uses
  tilt index `t/(N-1)`.
- CubicGrid: interpolating einspline B-spline, natural BCs, EXTRAPOLATES
  outside [0,1]; axes with dim==1 are dropped (coordinate ignored).
- LinearGrid4D: quadrilinear with **confirmed asymmetric** boundary handling
  (lower boundary linearly extrapolates via the negative fraction after
  truncation-toward-zero; upper boundary collapses to the final node). Pinned
  by the M0b golden; do not assume symmetric clamping.
- `size_rounding_factors` are runtime state (from image header / explicit
  integer dims), never recoverable from XML alone.
- `MagnificationCorrection` does not enter the position model (Fourier-only).

## AreTomo3 tilt-series model (reference: `GCorrPatchShift.cu`, `CFitPatchShifts.cpp`)

Units: motion-corrected tilt-series pixels, centered origin. `-AtBin` never
affects `.aln` values.

```
fX = Cx cos(TILT) - Cz sin(TILT)
u0 = fX cos(ROT) - Cy sin(ROT);  v0 = fX sin(ROT) + Cy cos(ROT)   # "Coord" frame
w_p = [Good_p >= 0.9] * exp(-100 * ((u0-CoordX_p)/Nx)^2 + ((v0-CoordY_p)/Ny)^2)
raw = (u0, v0) + sum(w s)/sum(w) + (TX, TY) + N/2
```

- `.aln` local `CoordX/Y` are **per-tilt projected patch-center positions**
  (excluding global shift), not static centers. `ShiftX/Y` are residuals.
- The IDW field is evaluated at the rotated, PRE-global-shift coordinate.
- Local tilt index `t` runs over the dark-removed, tilt-sorted list; `SEC`
  (1-based) maps back to raw stack sections.
- `TILT` already includes `AlphaOffset` — never apply it twice.

## Cross-tool mapping (constants in `conventions.py`)

| Quantity | Relation |
|---|---|
| tilt angle | `TILT = -(Angle + LevelAngleY)` (`WARP_TILT_ANGLE_SIGN = -1`) |
| tilt axis | `ROT = +TiltAxisAngle` (`WARP_TILT_AXIS_SIGN = +1`) |
| shifts | `TX/TY` (px, centered, post-rotation) <-> `AxisOffsetX/Y` (A) via the cryoet-alignment closed form |
| volume Z | `Cz_fit = -z_centered / s` (`ARETOMO_FIT_Z_SIGN = -1`); XY centers/orientation coincide |
| pixel centering | `x_canonical_px = u_centered + N/2 + HALF_PIXEL_OFFSET`; `HALF_PIXEL_OFFSET = 0.0`, provisional until the synthetic odd/even + kernel tests, then a fixed convention |
| local shift sign | AreTomo TS locals are positions ADDED; Warp GridMovement is SUBTRACTED |
| motion shifts | MotionCor and Warp movie both SUBTRACT corrections (`raw = x - S(x) - glob`); the sign flip exists only TS-vs-motion |

### Level angles vs Alpha/Beta

- `LevelAngleY` = `AlphaOffset` in meaning (constant stage-tilt offset), but
  AreTomo bakes it into the TILT column. Conversion is exact both ways via the
  TILT relation above; w2a records `AlphaOffset = -LevelAngleY` in the header
  for provenance only.
- `LevelAngleX` = `BetaOffset` in meaning (pitch about the perpendicular
  in-plane axis), but BetaOffset NEVER enters AreTomo's projection geometry
  (CTF-only). v1 policy: preserve BetaOffset from `--template-aln` if present,
  else write 0; record LevelAngleX in provenance. Its sign is NOT identifiable
  from projection residuals — no geometric calibration is attempted.

## Tilt matching (.aln <-> Warp XML)

1. Drop AreTomo dark frames (header `DarkFrame` lines) and Warp
   `UseTilt=false` tilts.
2. Compare stage angles: AreTomo `TILT - AlphaOffset` vs Warp `-Angle`.
3. Nearest-angle matching with 0.5 deg tolerance; require a bijection; fall
   back to SEC ordering on duplicate angles; hard-fail with a report if
   ambiguous.
4. Store the permutation. All store arrays are indexed in **Warp file order
   including darks**, with validity masks.

## Handedness (AreTomo3 vs Warp)

Empirically known: for tomograms of the same hand, either the tilt-axis angle
must be shifted by -180 deg on the AreTomo side, or the tilt-angle sign must
be inverted on the Warp side (see the reconstruction-hand comparison
screenshot at the workdir root). arewarpion's calibration reproduces and
explains this:

- The AreTomo global projection `fX = Cx cos(theta) - Cz sin(theta)` is
  EXACTLY invariant under jointly flipping `(theta, z) -> (-theta, -z)`.
  A tilt-angle sign flip alone is projection-visible; combined with a volume
  z-flip it is invisible and changes only the reconstructed hand. The
  axis-180 variant is the same correction composed with an in-plane 180 deg
  rotation (absorbed by each tool's alignment).
- Real-data calibration (warp_trial `24jul16a_Position_30_2`, independent
  alignments): with Warp's stored angles = -(aln TILT) — the angle-sign
  variant of the known fix — the projection models agree at 24.8 A RMS;
  all axis-mirror / axis-180 / z-flip candidates are 390-3700 A (margin
  0.064). Fed IDENTICAL metadata instead, the two tools' volumes differ by
  the projection-invisible joint flip = opposite hands, exactly as observed
  in the screenshot; `ARETOMO_FIT_Z_SIGN = -1` encodes that relative
  z-inversion.

## Sign calibration

`calibrate_signs(warp_xml, aln)` enumerates candidate sign tuples and requires
the configured tuple to be the argmin of global-model RMS **with a meaningful
margin** over the runner-up; zero-shift or symmetric datasets are rejected as
calibration input. The half-pixel convention is NOT part of calibration.

## Anisotropic pixels

Rejected explicitly at load in v1. (If ever supported: store
`(pixel_size_x, pixel_size_y)` and report per-axis residuals.)

## Dtypes

Fits: float64 through pure arewarpion operators. Compatibility evaluation of
emitted files: float32 (both tools are float32). Both recorded in fit
metadata.
