# RELION 5: projection, motion and CETS

For worked commands, see [native → CETS](../tutorials/relion-to-cets.md)
and [CETS → native](../tutorials/cets-to-relion.md).

`cets-nonrigid` supports RELION tomography (`relion`) and per-micrograph frame motion
(`relion-motion`). Tomogram projection, particle coordinates, image deformation and
movie motion have distinct conventions; one Euler or center rule cannot replace all
of them. This describes the implemented models and their CETS boundary.

## Tomogram projection

RELION's homogeneous projection matrices map corner-origin, bin-1 tomogram voxel
coordinates to corner-origin tilt-image pixels. Let `s` be
`rlnTomoTiltSeriesPixelSize`, `n_v` the bin-1 volume dimensions and `n_i` the image
dimensions. The stored tilt angles construct standard right-handed rotations:

```text
R_t = Rz(zrot_t) @ Ry(ytilt_t) @ Rx(xtilt_t)
c_v = floor(n_v/2)
c_i = (floor(nx/2), floor(ny/2), 0)
a_t = (xshift_angstrom, yshift_angstrom, 0) / s

P_t = [R_t | c_i + a_t - R_t*c_v]       # top three rows of the homogeneous matrix
q_native(X) = s * (P_t * [X/s, 1]).xy   # X and result in corner-origin Å
```

The projection uses integer-division centers. Explicit matrix imports must have a
proper rotation and valid homogeneous structure; scale, shear and reflections are
rejected. Pixel size belongs in coordinate conversion, not a non-rotation global.

Particle coordinates use another center rule. For modern centered coordinates,
the effective static point in corner-origin Å is:

```text
X_eff = centered_coordinate_angstrom - A_subtomogram * origin_angstrom + s*n_v/2
```

Legacy voxel coordinates use `X_eff = s*coordinate_px - A_subtomogram*origin_angstrom`.
`A_subtomogram` follows RELION's particle Euler convention, as implemented by
`angles_to_matrix3`; it is not the tilt projection's `R_t`. The modern particle
center `n_v/2` is floating-point. On odd dimensions it differs from both the
projection center and CETS's physical origin.

## Trajectories and image deformation

For a bound particle `i`, the complete observable applies its 3D trajectory before
projection and the per-image deformation afterwards:

```text
X_eff[i] -- add trajectory[t,i] --> 3D position at tilt t
    -- P_t projection --> image pixel position
    -- add image deformation at that position --> final image pixel position
    -- multiply by s --> q_native[t,i] in Å
```

Trajectory offsets are additive Å. Supported image deformation models are:

| Native model | Implemented behavior |
|---|---|
| linear | Three-coefficient additive field evaluated about the floating image center |
| spline | Bicubic Hermite field with values, X/Y slopes and twist at each node |
| Fourier | Native half-plane frequency enumeration and complex coefficient layout |

These are deformation terms even where their action is affine or spatially
constant. They are not inserted into CETS G. Native deformation models can be read;
RELION output from the CETS API represents local projected observations through
particle trajectories rather than reconstructing the original coefficient model.

## Correspondence to CETS

CETS uses Å with origin at array index `floor(size/2)`. Let CETS frame centers in
corner-origin Å be `C_v` and `C_i`, and write the native projection as `[R_t | b_t]`:

```text
X = p_CETS + C_v
G_t(p_CETS) = (R_t*(p_CETS + C_v)).xy + s*b_t.xy - C_i
q_CETS = q_native - C_i
r_t = q_CETS - G_t(p_CETS)
```

The adapter writes G to `Alignment.projection_alignments`, referencing image IDs,
and binds the reference `Tomogram` through `reference_volume_id`. The global-only
baseline projects the effective static coordinate, with trajectories and image
deformations disabled. Thus r already includes the projected trajectory effect.

| Native information | CETS representation |
|---|---|
| Proper projection matrices / rigid angles and shifts | Global projection operators |
| Effective static particle coordinates | `PointSet3D` in the reference-volume frame |
| Particle identity, half-set and class | Stable point IDs and typed point attributes |
| Trajectories and image deformations | Projected residual at bound particles |
| Available 3D trajectory | Independent `displacement_3d` channel |
| Native static-coordinate CTF depth | Independent `ctf_depth` channel |
| Optics, per-tilt CTF and dose | Acquisition records and image/series metadata |

RELION's particle model is defined on its complete bound set. The adapter evaluates
that set before selecting training and held-out subsets; it does not call the native
model at invented grid points. CETS associates each block with point IDs and each
tilt row with `tilt_image_ids`. Native particle visibility affects fitting validity;
finite observations outside the FOV are still retained separately as available.

## Export, trajectory lift and CTF

`api.fit(bundle, "relion", ...)` requires particle-bound observations for non-rigid
output. Supply the particles at `to-cets` time for AreTomo3 or Warp sources. A sampled
grid does not establish a continuous field from which arbitrary new particle
trajectories can be recovered. `no_particles=True` supports global-only output.

Given the emitted target projection G, source displacement d if available, and
complete observation q, the trajectory lift uses the residual remaining after d:

```text
e_t = q_t - G_t(p + d_t)
u_t = d_t + R_t.transpose() * (e_t.x, e_t.y, 0)
p_out = p + c
motion_t = u_t - c
```

All quantities here are in consistent physical Å frames. Proper rotations make
`R_t.transpose()` the lift for the in-plane component. The resulting projected
position is q independently of gauge c; adding a projection of d again would double
count it. The emitted image positions are checked to the existing 0.001-pixel limit.

RELION calculates CTF depth from the **static** particle coordinate, not its
per-tilt trajectory:

```text
depth_t(X_static) = hand * defocus_slope * (R_t*(X_static - s*n_v/2)).z
```

The CTF center is floating-point even though projection uses integer centers. This
is why matching image trajectories alone does not guarantee matching defocus.
`lowest-dose` chooses c from the lowest-dose emitted row. `ctf-optimal` chooses c to
minimize the static-coordinate CTF-depth discrepancy, using the supported singular
directions and the validated fallback limits. It changes static positions and
trajectory gauge while preserving projected positions. Depth deviation is reported;
missing source depth is not treated as a successful comparison.

Export writes an ordinary RELION/py2rely project with tomograms, per-series STAR,
particles, motion and optimisation-set files as applicable. Complete optics, dose,
CTF and handedness are required by the relevant operation; geometry-only placeholder
CTF is an explicit option. Native snapshots are never reopened to supply context.

## Movie motion (`relion-motion`)

The per-micrograph STAR model uses original, unbinned movie pixels, scaled by
`rlnMicrographOriginalPixelSize`. A corrected-image coordinate maps to raw frame f as:

```text
z_f = (f + 1) - rlnMicrographStartFrame     # f here is zero-based
xn = x_px / width - 0.5
yn = y_px / height - 0.5

local_f = sum c[term,power] * term(xn,yn) * z_f^power
term in {1, xn, xn², yn, yn², xn*yn}; power in {1,2,3}
q_raw = x_px - global_shift_f - local_f
```

There are 18 coefficients per output axis and no time-independent polynomial term.
CETS `MovieAlignment` stores `-s*global_shift_f` as its per-frame Translation and
`-s*local_f` as sampled residual. Centering cancels for the global translation;
the local field is still evaluated in the original native image coordinates.

A `-9999` shift sentinel in either axis makes a frame unavailable. The native
model has a preceding-observed-shift fallback for evaluating that row, but CETS
keeps it excluded, with no frame alignment or projected observation. Ordered
`frame_ids` preserve identity. Dose, pre-exposure, voltage and available EER settings
are carried in core metadata or tool-specific provenance.

Non-unit `rlnMicrographBinning` is refused: the supported model does not mix binned
polynomial coefficients with unbinned global shifts.

## Implementation and checks

- [Projection, particle coordinates and depth](../../src/cets_nonrigid/models/relion_ts.py),
  [image deformations](../../src/cets_nonrigid/models/relion_deform.py), and
  [movie motion](../../src/cets_nonrigid/models/relion_motion.py).
- [Trajectory lift and gauges](../../src/cets_nonrigid/fit/relion_traj.py) and
  [CETS RELION export](../../src/cets_nonrigid/relion_export.py).
- [Projection/motion model checks](../../tests/test_r1_relion_models.py),
  [deformation checks](../../tests/test_r5b_deformations.py),
  [3D-aware lift checks](../../tests/test_displacement_relion_lift.py), and
  [native project extraction](../../tests/test_wp2_relion_binary.py).

See the [shared exchange profile](../exchange-profile.md) for payload shape,
precision, held-out sampling, masks and the observable-at-samples contract.
