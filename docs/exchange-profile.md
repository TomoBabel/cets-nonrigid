# CETS non-rigid exchange profile 0.1

This is a lossy, point-sampled transmission format. The round-trip invariant is
**observables at samples, not model identity**. Samples do not select an
interpolator between points. Export fits the target's native model and reports
its representation error on separate held-out observations.

## Ownership and frames

`Alignment.non_rigid_alignment` owns the non-rigid component.
`has_non_rigid_alignment` is a derived Python property and is never serialized.
Absent/null means no component; a present component may contain displacement values that are all zero. A tilt-series alignment references a `Tomogram` as its reference
volume, including geometry-only tomograms whose path is null.

CETS JSON owns identity, geometry, proper rotations/translations, acquisition,
optics, CTF and provenance. Physical coordinates are right-handed Å, with origin
at array index `floor(N/2)`. The ported native kernels retain their validated
corner coordinates internally; the boundary applies the shared codec's centre
deltas. Anisotropic spacing and non-rotation projection operators are rejected.

```text
Native inputs
    |
    +--> native rigid parameters --> shared codec --> CETS Alignment --> G(p)
    |
    +--> validated complete native model -----------------------------> q(p)
                                                                        |
                                                          r(p) = q(p) - G(p)
                                                                        |
                   CETS JSON <---------------- descriptor ------------ Zarr
                       |                                                |
                       +---------------------- G(p) + r(p) -------------+
                                                   |
                                     fit target globals and local fields
                                                   |
                                held-out report + native files/project
```

G is evaluated in float64 from the document. Residuals are stored in float32;
the component-wise serialization bound is `ULP32(r) + 8*eps64*max(1,|G|,|q|,|r|)`.
The projected residual already includes the projected effect of 3D displacement.
Never add that effect again. Independent displacement/depth channels constrain
volume fitting and CTF-aware trajectory lifting.

## Descriptor example

```json
{
  "id": "TS_01_warp",
  "tilt_series_id": "TS_01",
  "reference_volume_id": "TS_01_volume",
  "projection_alignments": [],
  "non_rigid_alignment": {
    "profile_version": "cets-nonrigid/0.1",
    "kind": "tilt-series-projection-residual",
    "tilt_image_ids": ["TS_01_0", "TS_01_1"],
    "payload_uri": "experiment.nonrigid.zarr",
    "payload_group": "alignments/unique-group-id",
    "sampling": {"kind": "grid", "grid_shape": [15, 15, 5]},
    "heldout": {"method": "scrambled-sobol", "seed": 20260828, "count": 675},
    "channels": {"displacement_3d": "present", "ctf_depth": "present"},
    "context_digest": "<64 hexadecimal SHA-256 characters>",
    "digest_version": 1
  }
}
```

This illustrates field placement, not a complete processing document: active
projection operators and image/reference entities must also be supplied.
`tilt_image_ids[t]` identifies payload row `t`; every series image appears once,
including excluded images. Acquisition order is a separate image property.
Exclusion is alignment-scoped: no projection operator means no G for that row.
Residuals are zero-filled and unavailable there. Dark/refined angles retain their
provenance through `TiltAngleObservation`; only proven stage angles populate
`nominal_tilt_angle`.

Particle sampling instead uses `{"kind":"particles","annotation_id":"picks"}`.
The referenced PointSet3D owns coordinates, stable point IDs and typed per-point
attributes. Training and held-out `point_ids` arrays select disjoint subsets of
that identity set. Particles must be supplied at sampling time for deformation-aware
RELION output; grid-only bundles cannot invent trajectories at new locations.

## Payload

One local Zarr v3 store accompanies each CETS dataset document. It contains a
separate immutable payload group for every non-rigid alignment across all regions.
URIs resolve relative to the JSON directory; absolute local paths and local file
URIs are supported. Remote Zarr access is not implemented in this profile.

### Multiple regions and alignments

All regions in a dataset document share the store. Each non-rigid alignment has
its own group, so two alignments in one region and an alignment in another region
remain separate:

```text
experiment.cets.json
  regions
    region_A
      alignments
        alignment_A1 --> alignments/<group-A1>
        alignment_A2 --> alignments/<group-A2>
    region_B
      alignments
        alignment_B1 --> alignments/<group-B1>

experiment.nonrigid.zarr/
  alignments/
    <group-A1>/       points, projected_residual, masks, heldout/, ...
    <group-A2>/       points, projected_residual, masks, heldout/, ...
    <group-B1>/       points, projected_residual, masks, heldout/, ...
```

Each alignment's `non_rigid_alignment` descriptor uses the same
`payload_uri: experiment.nonrigid.zarr` and a distinct `payload_group`, such as
`alignments/<group-A1>`. The group labels above are schematic; the writer derives
actual group IDs from `(region_id, alignment_id)`.

Groups can have different image counts, sample counts, reference volumes and
optional channels. Their geometry stays in the corresponding CETS entities;
arrays are not concatenated across regions. Movie alignments follow the same
arrangement under `movies/`. Alignments without a non-rigid component need no
payload group. A document with no sampled deformation needs no Zarr store.

The one-store requirement is a profile 0.1 policy, not a limitation of CETS or
Zarr. The current reader rejects a document whose alignment descriptors resolve
to different stores; separate stores per region would require a profile and
reader change.

### Arrays within an alignment

```text
experiment.cets.json
  Alignment.non_rigid_alignment ----> experiment.nonrigid.zarr/
                                       alignments/<unique-id>/
                                         points                 (N,3)   float64
                                         projected_residual     (T,N,2) float32
                                         sample_valid           (N,)    bool
                                         observation_valid      (T,N)   bool
                                         projection_valid       (T,N)   bool
                                         weights                (T,N)   float32
                                         displacement_3d        (T,N,3) float32, optional
                                         displacement_valid     (T,N)   bool, optional
                                         ctf_depth              (T,N)   float32, optional
                                         ctf_depth_valid        (T,N)   bool, optional
                                         point_ids              (N,)    UTF-8, particles
                                         heldout/               same arrays
                                         reports/               diagnostics
                                         snapshots/             native bytes + hashes
```

Finite outside-image observations remain available even when fitting validity is
false. Weights never encode validity. Optional channels have independent masks;
unavailable entries are zero-filled. Displacement states are `present`,
`zero_at_samples` and `none`; depth states are `present` and `none`. Zero at samples
makes no claim between samples. Incomplete optional channels remain representable;
fitters requiring complete support refuse them explicitly.

Attrs repeat only profile, digest, owner identity, units and dimension names.
They never duplicate global arrays, image sizes, spacing or row metadata. Native
snapshots are provenance: fitting never reopens them. A `snapshot:<role>` artifact
reference addresses the matching role in that alignment's snapshots group.

A missing/empty held-out set reports `not_evaluated` and null metrics, never a pass.
The block still exists. Grid sampling includes native borders; defaults are
15×15×5 (tilts) and 11×11 (movies). Grid held-out points are scrambled Sobol;
particle holdout uses the validated deterministic split and may be empty for
small sets.

## Global/local classification

| Source | Global G | Residual payload |
|---|---|---|
| AreTomo3 | ROT, TILT, TX/TY | Local patch field |
| Warp tilt series | Angles, LevelAngle, AxisAngle/Offset | Movement and volume grids, including constants |
| RELION tomography | Proper projection rotations and shifts | Image deformation and particle trajectories |
| Warp movie | Temporal GridMovement | GridLocal and pyramids |
| MotionCor/.mcaln | Per-frame global shift | Patch field |
| RELION motion | Per-frame shift | Local polynomial |

AreTomo TILT already contains AlphaOffset. BetaOffset is CTF/provenance metadata,
not another projection rotation. Warp GridAngle, MagnificationCorrection and
non-unit runtime SizeRoundingFactors remain outside the validated model.

The rigid Warp adapter folds constant grids; this package deliberately preserves
them in the deformation component for tilt series. Compare total observations
when testing that overlap. Its folded rigid G equals G+r only for the compatible
spatially constant cases, not for a general local field. Constant volume warps
also retain 3D and CTF-depth information that a rigid shift cannot express.

## Movies

`MovieStack` supplies shared geometry. Its `MovieAlignment` owns ordered frame IDs,
per-frame `Translation` operators, gauge/reference frame, provenance and an optional
non-rigid descriptor. Points are 2D corrected/reference image coordinates, and
`G_f(x)+r_f(x)` gives their sample positions in raw frame f. Integration spans and
excluded frames are preserved separately from payload row order. RELION motion
supports missing-frame sentinels; targets with no exclusion flag may synthesize
values on excluded frames, which are identified in the fit report.

## Binding and publication

Digest v1 hashes sorted compact UTF-8 JSON of context identity, ordered row IDs and
activity, folded operators, image/reference frames, acquisition/CTF conventions,
sampling descriptors and particle binding. Finite floats use exact float64 hex
strings and normalize negative zero. File paths and descriptive provenance are
excluded. See `CetsContext.digest_record` and `canonical_json` for the executable
algorithm. A context mismatch is refused; changed rigid geometry requires resampling.

Attachment returns a copied document. Sampling checks that its G agrees with the
native rigid-only baseline on training and held-out points at a precision-scaled
bound. Existing Warp alignments with folded constants need a new alignment instance.
Publication reserves a new store, writes and validates temporary groups, renames
those groups, marks the store complete and publishes JSON last without replacement.
Existing outputs are refused. Fit reports are bound to the source context digest
and are saved as a new bundle version.
