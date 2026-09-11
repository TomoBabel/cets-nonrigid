# Implemented CETS additions

See the [core-schema checklist](core-schema-checklist.md) for upstream adoption,
remaining feature-branch gaps and converter integration work.

The preserved cets-data-models checkout is unchanged. These additions live in the
independent cets-data-models-dev clone and the published
[feature/nonrigid-alignment branch](https://github.com/TomoBabel/cets-data-models/tree/feature/nonrigid-alignment).
The numerical package imports its ordinary cets_data_model types and references
that branch in pyproject.toml. The changes are not merged into main; no PR has been
opened for this feature branch. The original cets-data-models-proposals.md is unchanged.

| Runtime need | Core definition | Status |
|---|---|---|
| Global and local alignment ownership | Alignment.non_rigid_alignment; derived has_non_rigid_alignment | Implemented locally |
| Shared reference frame | Alignment.reference_volume_id → Tomogram; Tomogram.alignment_id | Implemented locally |
| Ordered payload rows | NonRigidAlignment.tilt_image_ids; MovieAlignment.frame_ids | Implemented locally |
| Frame motion | MovieStack image geometry; MovieFrame identities/integration spans; MovieAlignment; FrameAlignment Translation | Implemented locally |
| Versioned sampled payload | NonRigidAlignment profile/kind/URI/group/digest; discriminated GridSampling/ParticleSampling; HeldoutSampling; NonRigidChannels | Implemented locally |
| Dark rows and angle provenance | Alignment.exclusions; TiltAngleObservation; image acquisition order/dose | Implemented locally |
| Particle binding | PointSet3D.point_ids and typed PointAttribute columns | Implemented locally |
| Scientific acquisition context | Instrument and AcquisitionSession, adopted from PR #35 | Anticipatory integration |
| CTF conventions | Degree angle/phase units; nullable series handedness/slope; preservation of unknown legacy handedness | Implemented locally, 7458f7d |
| Native CTF diagnostics | CTFMetadata.fit_score and fit_resolution (Å) | Implemented locally, 54a46d8 |
| Provenance without a parallel metadata model | ProcessingProvenance, NativeParameter, NativeArtifact | Implemented locally |
| Reproducible generated public types | PR #32 generator/mixins, PR #34 coordinate conventions, JSON-Schema-visible constraints | Anticipatory integration plus documented corrections |
| Cross-entity checks | utils.references.validate_document_references | Implemented locally |

Exact commits and adopted PR heads are recorded in the development schema's
[ledger](https://github.com/TomoBabel/cets-data-models/blob/feature/nonrigid-alignment/docs/schema-change-ledger.md). Environment locks
record the feature revision used by numerical validation.

A payload's observation_valid mask records availability independently of the
projection_valid fitting mask. Finite out-of-field observations are retained
because the validated trajectory lift uses them. Optional 3D displacement and
CTF depth have their own availability masks. These are array-level profile rules,
not duplicated geometry or core entity properties.

```text
Dataset
  instruments[] <---- acquisition_sessions[]
  regions[]
    tilt_series[] --images--> TiltImage (physical frame, acquisition, CTF)
    tomograms[] <-------------------+
    alignments[]                   | reference_volume_id
      projection_alignments[] --> TiltImage.id
      non_rigid_alignment --------> Zarr group (points, residuals, channels)
    annotations[] --> Tomogram.id    ^
      PointSet3D.point_ids ----------+ particle sampling
    movie_stack_collection
      movie_stacks[].stacks[]
        MovieStack (shared image frame)
          images[] (MovieFrame.id)
          alignments[] (MovieAlignment)
            frame_alignments[] --> MovieFrame.id
            non_rigid_alignment --> Zarr group
```

After the upstream PRs merge, reconcile only the development clone, replace
anticipatory commits with upstream equivalents, update the exact dependency lock,
and rerun generation, schema, codec, converter and numerical gates.
