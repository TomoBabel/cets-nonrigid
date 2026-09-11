# Warp: projection, motion and CETS

For worked commands, see [native → CETS](../tutorials/warp-to-cets.md)
and [CETS → native](../tutorials/cets-to-warp.md).

`cets-nonrigid` supports Warp tilt-series XML (`warp`) and movie XML
(`warp-movie`). Tilt projection uses the required, pinned `warpylib` backend.
Movie motion uses the ported model checked against Warp's C# implementation.
The two models give `GridMovement` different roles in the CETS global/local split.

## Tilt-series coordinates and projection

Warp's model takes reference-volume points in corner-origin Å and returns
sampling positions in raw motion-corrected tilt images, also in corner-origin Å.
Let `P` be a volume point, `V` the volume extent and `I` the image extent in Å.
Using standard right-handed column-vector rotations, the implemented chain is:

```text
D_t = (Dose[t] - min(Dose)) / (max(Dose) - min(Dose))
d_t = GridVolumeWarp(Px/Vx, Py/Vy, Pz/Vz, D_t)

R_t = Rz(+AxisAngle[t]) @ Ry(-(Angles[t] + LevelAngleY)) @ Rx(+LevelAngleX)
z_t = R_t @ (P - V/2 + d_t)
b_t = z_t.xy + (AxisOffsetX[t], AxisOffsetY[t]) + I/2

q_native(P) = b_t - GridMovement(b_t.x/Ix, b_t.y/Iy, t/(T-1))
```

Angles are in degrees. The singleton index axis uses coordinate zero. Volume
warping is evaluated at the undeformed volume point; movement is evaluated at
the **premovement image position** `b_t`. Moving either lookup to the corrected
image position changes the model.

| Field | Units and interpolation | Meaning |
|---|---|---|
| AxisOffsetX/Y | Å | Global post-projection image shift |
| GridVolumeWarpX/Y/Z | Å; quadrilinear `LinearGrid4D` | 3D displacement indexed by position and normalized dose |
| GridMovementX/Y | Å; cubic `CubicGrid` | Subtracted 2D image correction indexed by premovement position and tilt index |

Cubic grids use the validated einspline interpolation and extrapolation;
singleton dimensions ignore that coordinate. The 4D grid has native asymmetric
boundary handling, which must not be replaced with symmetric clamping. Dose and
tilt index are distinct coordinates. Dose normalization uses all native rows,
including excluded tilts. All-equal dose is not a usable placeholder for Warp:
the loader, synthesis and fitting/writing gates refuse undefined dose metadata.

## Correspondence to CETS

CETS centers physical coordinates at array index `floor(size/2)`. Set
`C_v = floor(volume_size/2)*volume_spacing` and
`C_i = floor(image_size/2)*image_spacing`. The boundary evaluates:

```text
p_CETS -- add C_v --> P_Warp -- full Warp chain --> q_native -- subtract C_i --> q_CETS
    |                                                                              |
    +------------------ evaluate document G ----------------------------------------+
                                                                 r = q_CETS - G(p)
```

`Alignment.reference_volume_id` selects the shared `Tomogram` frame. Image
`array_to_physical` transforms supply spacing and centering. The shared codec
includes the `N/2` versus `floor(N/2)` center differences; an odd axis can therefore
contribute a half-pixel center delta. No manual half-pixel correction is added to
samples after the codec has established the frames.

For tilt series, G contains only Angles, LevelAngleX/Y, AxisAngle and AxisOffsetX/Y.
**Every movement and volume-warp grid remains deformation**, even a spatially
constant grid or a per-tilt `1×1×T` movement grid. Import constructs G through the
shared rigid codec using a private copy with those grids zeroed, while q uses the
original full model.

This differs intentionally from the rigid Warp reader, which folds constant grids
into its shift. Attaching to that folded alignment fails the native-baseline gate;
create a compatible new alignment. For suitable constant fields, compare its folded
G with this package's G+r. A general spatially varying field cannot be represented
by a rigid-only adapter, and G alone is not necessarily Warp's best rigid approximation.

Payload rows follow explicit `tilt_image_ids`, not dose order. `UseTilt=false`
keeps the image and angle provenance but supplies no global projection or projected
observation. Warp Angles remain an unknown-kind angle observation unless the adapter
can establish stage-angle provenance; they are not automatically nominal stage angles.

## Displacement and CTF depth

The payload can retain `d_t` separately as `displacement_3d` and the signed defocus
contribution as `ctf_depth`. The projected residual already includes the projected
effect of `d_t`. Reconstruct q as **G+r**, never G+r plus another projected displacement.

Warp computes CTF depth from the warped centered point. With normal angle handedness
it is `z_t.z`. With `AreAnglesInverted`, the depth path flips the warped point's Z
and the tilt/level-X rotations; the XY projection path remains the one above.
The adapter evaluates that native depth path rather than assuming a generic sign
flip. Warp adds depth in Å converted to µm to its per-tilt defocus.

Core per-image CTF metadata carries defocus in Å and angles/phase in degrees;
series metadata records handedness. A constant volume warp remains meaningful to
3D coordinates and CTF depth even if its XY effect could be folded into a shift.
The optional channels have independent masks and availability.

## Import and target fitting

```text
Warp XML + resolved geometry/acquisition metadata
    -> warpylib full model + native rigid parameters
    -> CETS G + sampled residual + optional 3D/depth channels
    -> reconstructed observations G+r
    -> target globals
    -> optional target volume-warp fit
    -> target movement fit at target premovement positions
    -> held-out diagnostics + native XML / Warp project
```

`api.to_cets("warp", ...)` discovers metadata where available and requires missing
physical dimensions/pixel size to be supplied explicitly. Native q runs at the
backend's float32 precision; G is evaluated from the CETS document in float64.

`api.fit(bundle, "warp", volume_warp_grid=...)` fits the optional 3D channel first.
The movement fitter then subtracts the target's own premovement projection, so it
fits only the remaining 2D correction. Without a volume-warp fit, image motion can
approximate projected observations while losing 3D/depth behavior; diagnostics
report the available comparison. Incompatible target image/reference geometry is
refused. Project export writes XML, tomostar/settings and image assets from CETS
context and explicit inputs, without reopening native provenance snapshots.

## Movie motion (`warp-movie`)

Movie mapping goes from a corrected/reference image point `x` to its sample
position in raw frame `f`. All grid values are in Å:

```text
u = (x.x/Ix, x.y/Iy)
tau = f / max(1, F-1)
tau_fractional = tau * FractionFrames

q_f(x) = x - GridMovement(u, tau_fractional)
           - GridLocal(u, tau_fractional)
           - sum_k PyramidShift_k(u, tau)
```

The pyramid time coordinate is not scaled by FractionFrames. Dimensions, frame
count and FractionFrames are runtime context that movie XML alone does not supply.
The native model evaluates fields at the input corrected coordinate, not at an
already shifted point.

For this pipeline the temporal-only `GridMovement(1,1,F)` is **global drift**.
`MovieAlignment.frame_alignments` stores its negative as an Å translation.
GridLocal and pyramid terms form the sampled residual. This is deliberately
different from tilt-series GridMovement. A spatially varying movement grid cannot
serve as a per-frame translation and fails the baseline check.

`MovieStack` holds geometry, `frame_ids` establishes row order, and the alignment
stores its gauge. Recentring into CETS leaves a pure translation unchanged because
input and output use the same image frame. A point falling outside the image does
not mean the entire frame is excluded.

## Limits and implementation references

GridAngle*, magnification correction, anisotropic pixels and non-unit runtime
SizeRoundingFactors are outside this model. Equal-dose refusal and known unsupported
terms remain explicit; no new native correction is inferred from missing metadata.

- [Tilt wrapper and 3D/depth paths](../../src/cets_nonrigid/models/warp_ts.py),
  [movie chain](../../src/cets_nonrigid/models/warp_movie.py), and
  [CETS rigid adapter](../../src/cets_nonrigid/native.py).
- [Volume and movement fitting](../../src/cets_nonrigid/fit/warp_ts_fit.py),
  [displacement checks](../../tests/test_displacement_model.py), and
  [movie golden checks](../../tests/test_m6_warp_movie_model.py).
- [CETS constant-grid and quantization checks](../../tests/test_cets_exchange.py).

See the [shared exchange profile](../exchange-profile.md) for storage, integrity,
availability masks and the lossy observable-at-samples contract.
