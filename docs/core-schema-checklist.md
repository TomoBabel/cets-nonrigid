# Core-schema checklist: cets-nonrigid

Audited 2026-09-10 against upstream `cets-data-models` main at
`5e2ba016047f0d9635c3670332a9bf945dfdf339` and
[feature/nonrigid-alignment](https://github.com/TomoBabel/cets-data-models/tree/feature/nonrigid-alignment)
at `54a46d8c58b473fdb4eef839f19427911c01b82f`. Remote heads were checked on that date.
The existing rigid converters pin `b415e95` (PR #34), which is ahead of main for
coordinate conventions but lacks the non-rigid feature additions.

An unchecked item means work remains before the information is exchanged through
upstream CETS. **Branch** means the schema is implemented on the feature branch,
not merged into main. **New schema** means it is missing there too. **Convention**
means existing transform/metadata types need a defined contract. **Converter** means
an available core field still needs reader/writer support. These are documentation
checklists, not claims that the pending changes have been implemented.

The goal is to transmit scientific observations and required metadata using the
CETS document and its declared payload. The sampled format remains lossy:
**observables at samples, not model identity**. Original native files may be retained
as artifacts; byte-identical XML/STAR or original spline coefficients are a separate
archival concern. An artifact or a dropped-information message does not constitute
working support for a scientific quantity.

## 1. Bring the implemented feature schema into upstream CETS

These items have implementations on the feature branch; their upstream merge and
reconciliation are pending. See [implemented additions](cets-additions.md).

- [ ] **C01 — Branch: canonical image/projection contract.** Adopt the PR #34
  coordinate names/units and three-entry projection sequence with reproducible
  public model generation. Keep floor(N/2) centering, proper rotation operators
  and explicit `tilt_image_id`. Main's two-entry limit is insufficient for the
  existing codec's three-entry output. A fourth X-tilt step is not required:
  `Ry(tilt) @ Rx(x_tilt)` is already representable in an affine rotation block.
- [ ] **C02 — Branch: alignment identity and frame binding.** Merge optional
  `Alignment.id`, `name`, `reference_volume_id` and `Tomogram.alignment_id`.
  The reference is an ordinary Tomogram, with a nullable path. Validate references
  without requiring IDs in previously valid rigid documents.
- [ ] **C03 — Branch: first-class sampled deformation.** Merge
  `Alignment.non_rigid_alignment`, the derived nonserialized presence property,
  profile/kind/URI/group/digest fields, row IDs, grid/particle sampling, held-out
  descriptors and channel states. Keep masks, sample arrays and fit weights in the
  declared payload; do not duplicate global geometry in Zarr or add a second
  serialized presence boolean.
- [ ] **C04 — Branch: acquisition and angle/activity semantics.** Merge
  `acquisition_order` (zero-based), `exposure_dose` (e-/Å²), `exposure_time`,
  `TiltSeries.nominal_tilt_axis_angle`, `collection_metadata_path`, alignment-scoped
  exclusions and `TiltAngleObservation`. Define nominal as acquired stage angle,
  preserve unknown/refined angles, and distinguish explicit exclusion from a
  generally incomplete document. `accumulated_dose` already exists; its exclusive
  pre-exposure meaning must remain explicit. Never recover the last exposure by
  subtracting successive pre-exposures.
- [ ] **C05 — Branch: authoritative optics.** Merge Instrument/AcquisitionSession
  and series references: voltage in kV, Cs in mm and amplitude contrast as a fraction.
  Do not put duplicate optics on every image. Session dose rate is e-/Å²/**second**;
  the legacy companion's `dose_rate` and movie CLI's similarly named dose per frame
  must map to `exposure_dose`, not be copied into that session field.
- [ ] **C06 — Branch: CTF semantics and diagnostics.** Merge explicit degree units
  for angle/phase, nullable series handedness and slope, and `CTFMetadata.fit_score`
  / `fit_resolution` (Å). The old per-image handedness field already exists;
  reconcile its legacy default and conflicts with series values without turning
  unknown into -1. Numerical mappings of native handedness need source-specific tests.
- [ ] **C07 — Branch: movie identities, geometry and alignment.** Merge shared
  MovieStack geometry, `raw_frame_count`, MovieFrame IDs/source integration spans,
  MovieAlignment/FrameAlignment, ordered frame IDs, gauge/reference frame and
  non-rigid descriptor. Carry corrected-image → raw-frame map direction and use
  absence of a frame alignment only according to the processing profile's rules.
- [ ] **C08 — Branch: particle identity and attributes.** Merge `PointSet3D.point_ids`
  and typed `PointAttribute` columns, including the relation to particle sampling.
  Validate equal lengths, unique identities and coherent training/held-out subsets.
- [ ] **C09 — Branch: processing provenance.** Merge ProcessingProvenance on
  Alignment/MovieAlignment/Tomogram, with software/version, NativeParameter,
  NativeArtifact and warning/loss records. Ordinary core imports must remain free
  of torch, warpylib and cets-nonrigid.

## 2. Gaps still present on the feature branch

The names below are design proposals, not current API fields.

- [ ] **N01 — New schema: oriented-point identity and metadata.** Extend
  `PointMatrixSet3D` with the same IDs and attributes as PointSet3D, preferably via
  a shared point-set mixin. It currently has positions/matrices but lacks both new
  properties. Define integer/string/boolean/float and missing-value semantics for
  columns, plus optics-group references; arbitrary large integer IDs must not be
  coerced through float columns. Preserve half-set, class, group, particle IDs and
  scores when points are reordered. The numerical import/export must also retain
  particle orientations and all supported attributes; a schema change alone will
  not fix its current position-focused boundary.
- [ ] **N02 — New schema: annotation provenance and object identity.** Annotation
  currently has id/type/name/source_tomogram_id, not ProcessingProvenance or typed
  object/method/confidence/source metadata. Add those to support picks obtained
  from STAR, copick and Portal without hiding their scientific identity in an
  alignment-level JSON string. Include instance/group identity where provided.
- [ ] **N03 — New schema/convention: pixel-size roles.** Add an explicit CTF-fit
  sampling context (for Warp `CTF.PixelSize`, potentially different from image
  pixels). Acquisition movie geometry is already expressible on the branch's
  MovieStack, even without enumerating frames: define and populate that role.
  Keep reconstruction spacing in Tomogram and particle-coordinate spacing in the
  source annotation context; do not overload tilt-image `array_to_physical`.
- [ ] **N04 — New schema + model: spatial CTF fields.** Define sampled defocus U/V,
  astigmatism orientation and phase fields with frames, units, validity and baseline
  semantics. Scalar per-image CTF plus `ctf_depth` does not preserve arbitrary
  spatial `GridCTF*` variation. Both native evaluation and target fitting need
  independent validation before claiming support.
- [ ] **N05 — New schema + model: orientation and reconstruction weights.** Specify
  per-particle/per-tilt rotation observations for Warp `GridAngle*`, and separate
  dose/location B-factor and weighting channels if these are to be transmitted
  scientifically. They are currently outside the validated model. Payload `weights`
  are fit weights, not reconstruction weights or a place to store these fields.
- [ ] **N06 — Convention/profile + model: file-grid geometry.** Define file-axis
  permutation/reversal, direction and origin offsets for reconstructed tomograms
  (`FlipVol` layouts and engine-specific grid offsets), plus an explicit relation
  between reconstructed and alignment-reference grids. Existing MapAxis/Affine/
  Translation types are starting points; validate readers and helpers for their
  agreed composition. Recording FlipVol alone does not register the grid.
- [ ] **N07 — Convention/profile + model: non-unit runtime scales.** Define how
  anisotropic image sampling / native SizeRoundingFactors and magnification
  correction enter the observation model if support is added. Core Scale/Affine
  already exist; current isotropic/proper-rotation processing rules and native
  kernels are the restriction. Do not claim a generic affine field alone fixes it.
- [ ] **N08 — New schema, where shared semantics are needed: scientific context.**
  Define estimated specimen thickness (Å, distinct from reference-box depth),
  processing/reconstruction method, and typed per-field measurement provenance
  when consumers need these as interoperable facts. Native-only Thickness,
  AlphaOffset/BetaOffset, EER options and gauges can already be retained as scoped
  NativeParameters once C09 is adopted; promote only concepts with a shared meaning.

## 3. Converter work that is not an additional schema field

- [ ] **I01 — Map every required native/companion value.** Use C04–C09 for core
  metadata, and NativeParameters/Artifacts for remaining native details. Capture
  native software versions when discoverable. Record which input supplied a value
  and whether a default was used. Unknown values must remain unknown.
- [ ] **I02 — Complete native detail preservation deliberately.** Audit `.aln`
  GMAG/SMEAN/SFIT/SCALE/BASE, Thickness, source angle offsets, Warp processing and
  CTF parameters, frame integration and EER settings, STAR flavour/columns/optics
  tables and raw particle gauge decomposition. Store noninterpreted source detail
  as provenance with stable entity IDs; fields affecting observations require a
  validated model, not merely a serialized value. Original artifacts remain archival.
- [ ] **I03 — Reconcile rigid-only Warp imports.** cets-warpm folds constant movement
  and volume grids; this package leaves tilt-series grids in deformation. A future
  converter integration must construct a new native-rigid alignment before attaching
  samples. Compare total projections, and separately retain 3D/depth semantics.
- [ ] **I04 — Maintain movie-specific splits.** Temporal Warp movie GridMovement is
  global; GridLocal/pyramids are local. Preserve temporal identity/integration and
  reference gauge through each writer; never apply the tilt-series rule to movies.
- [ ] **I05 — Migrate consumers and locks.** After upstream reconciliation, update
  the dependency lock and shared codec/converter mappings. Resolve conflicts between
  old companion values and core fields explicitly; core must be authoritative for
  new documents. Replace cets-warpm's frozen arewarpion oracle only in a separate,
  validated downstream change.

## 4. Evidence and completion checks

- [ ] Every companion field or native datum has a disposition: core field,
  core-scoped native provenance, derived value, separately declared artifact, or
  explicitly unsupported scientific behavior. A loss message is not preservation.
- [ ] JSON Schema, generated Pydantic models and reference validation agree;
  regeneration is reproducible and legacy rigid examples remain valid.
- [ ] Export operates with companion files and native snapshots unavailable,
  using only CETS plus explicitly supplied image assets. No missing metadata is
  silently reconstructed from the discarded native files.
- [ ] Check multiple alignments/reference tomograms, dark rows, reordered images
  and particles, unequal final exposure, unknown handedness, distinct acquisition/
  image/CTF pixels and nontrivial movie integration. Retain existing numerical
  tolerances and independent goldens.
- [ ] New scientific channels receive independent native-model checks and a new
  profile version where required; lossy fits report held-out error and limitations.

Audit sources: [native boundary](../src/cets_nonrigid/native.py),
[metadata](../src/cets_nonrigid/metadata.py), [discovery](../src/cets_nonrigid/inputs.py),
[RELION export](../src/cets_nonrigid/relion_export.py),
[exchange profile](exchange-profile.md), and the shared codec's
[companion model](https://github.com/uermel/cryoet-alignment/blob/2eebafb7ebb641b07befa8eba20958a0870c6768/src/cryoet_alignment/io/cets/companion.py).
The corresponding cets-aretomo3 and cets-warpm repositories each contain a
`docs/core-schema-checklist.md` with their specific mappings and acceptance checks.
