# Tutorial: RELION 5 → CETS

Sample a RELION tomogram at its bound particles, including trajectories and image
deformation. See [RELION conventions](../pipelines/relion.md) for projection,
particle-center and CTF-depth rules.

## Inputs

Activate the [package environment](../../README.md#development-installation).
This example selects tomogram `TS_01` from a native project:

```text
input/relion/
    optimisation_set.star
    tomograms.star
    particles.star
    motion.star                 when trajectories are present
    tilt_series/TS_01.star
    ... referenced images
```

Keep STAR-internal paths resolvable from the native project root. The optimisation
set must identify the intended tomograms, particles and optional motion STAR.
This import requires the selected tomogram's particles; it does not invent a
continuous deformation on a grid.

## Import one tomogram

Replace the example's 4096 × 4096 raw tilt-image dimensions with your own.
`--image-size` is not the particle extraction box. Pixel size and volume dimensions
come from RELION metadata.

```bash
cets-nonrigid to-cets relion input/relion/optimisation_set.star \
  -o exchange/TS_01-relion.cets.json \
  --tomo-name TS_01 --image-size 4096x4096
cets-nonrigid validate exchange/TS_01-relion.cets.json
cets-nonrigid inspect exchange/TS_01-relion.cets.json
```

To select files explicitly, add `--particles-star input/relion/particles.star`
and, when applicable, `--motion-star input/relion/motion.star`.
For a nonstandard layout, `--config` accepts a `source.project_root` path;
configuration paths resolve from your current working directory.

The output is `exchange/TS_01-relion.cets.json` plus
`exchange/TS_01-relion.nonrigid.zarr`. The document owns frames, G, particles,
optics and CTF metadata. G is the static deformation-disabled projection. The
residual includes projected trajectories and image deformations; trajectories and
native CTF depth also have independently available channels.

Effective particle coordinates already include RELION origin/Euler corrections
before mapping into CETS. Do not apply those corrections again.

## Check the result and choose a target

Check the selected tomogram, image count and particle count. Training and held-out
IDs are subsets of the same bound particle set. Small sets may have no held-out
subset. `validate` checks references, geometry and payload integrity; export fits
produce numerical diagnostics.

Follow [CETS → Warp](cets-to-warp.md) for grid fitting,
[CETS → AreTomo3](cets-to-aretomo3.md) for patches, or
[CETS → RELION](cets-to-relion.md) for trajectory export. Sparse particles may not
support dense target grids; global representability still applies.

To import all tomograms, omit `--tomo-name` and use an output directory. Each
series gets its own document and payload; missing required particle/scientific
metadata causes reported batch failures.

## Movie motion

Use a per-micrograph motion STAR, not an aggregate micrographs table:

```bash
cets-nonrigid to-cets relion-motion input/relion/movie_001.star \
  -o exchange/movie_001-relion.cets.json --grid 11x11
cets-nonrigid validate exchange/movie_001-relion.cets.json
cets-nonrigid inspect exchange/movie_001-relion.cets.json
```

The file supplies original image dimensions, pixel size, frame shifts and local
polynomial coefficients. CETS stores negative global corrections as frame
translations and samples the local residual. Missing-shift sentinel rows remain
excluded. Non-unit micrograph binning is outside the supported model.

Keep JSON and Zarr together, and use new output names for repeated imports.
