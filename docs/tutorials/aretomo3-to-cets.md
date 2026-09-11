# Tutorial: AreTomo3 → CETS

Import a tilt-series `.aln` as CETS globals and sampled local residuals. Movie
motion is a separate example below. See [AreTomo3 conventions](../pipelines/aretomo3.md)
for projection equations and shift signs.

## Inputs

Activate the [package environment](../../README.md#development-installation).
Run from your working directory, replacing these example paths and values:

```text
input/aretomo3/
    TS_01.aln             rigid rows, dark rows and local patches
    TS_01_TLT.txt         raw-section angles, acquisition indices, per-image dose
    TS_01_CTF.txt         per-tilt CTF, when available
    TS_01.mrc             unaligned, motion-corrected tilt stack, when available
```

This example uses 1.54 Å tilt-image pixels and a reference box of
4096 × 4096 × 1000 **tilt-image pixels**. A reconstructed volume file is not
required. Convert a binned reconstruction's box to its physical extent before
choosing these dimensions. Adjacent native companions are discovered automatically.

## Import and check

```bash
cets-nonrigid to-cets aretomo3 input/aretomo3/TS_01.aln \
  -o exchange/TS_01-aretomo.cets.json \
  --pix 1.54 --tomo-size 4096x4096x1000 \
  --dose-per-tilt file --grid 15x15x5
cets-nonrigid validate exchange/TS_01-aretomo.cets.json
cets-nonrigid inspect exchange/TS_01-aretomo.cets.json
```

`--dose-per-tilt file` selects genuine per-image doses from acquisition metadata.
With mdoc acquisition order and a known constant exposure, use
`--mdoc-dir input/mdoc --dose-per-tilt 3.87` instead. Exposure is electrons/Å².
Tilt order can differ from acquisition order; a scalar dose cannot establish that
association. `--ctf-file` accepts an explicitly located CTF companion.

The outputs must be kept together:

```text
exchange/
    TS_01-aretomo.cets.json       geometry, globals, metadata, payload descriptor
    TS_01-aretomo.nonrigid.zarr   training/held-out points, residuals and masks
```

The document defines G; the payload stores r = q − G from the complete native
projection q. `--grid` sets sampling density, not target patch count. Held-out
points are generated separately. Dark images remain identified with unavailable
projections masked. Native snapshots are provenance, not subsequent fit inputs.

## Include particles for RELION

For later trajectory export, sample the intended particles during import:

```bash
cets-nonrigid to-cets aretomo3 input/aretomo3/TS_01.aln \
  -o exchange/TS_01-aretomo-particles.cets.json \
  --pix 1.54 --tomo-size 4096x4096x1000 --dose-per-tilt file \
  --particles input/TS_01-particles.txt
cets-nonrigid validate exchange/TS_01-aretomo-particles.cets.json
```

The text file has one `x y z` triple per line in Å from the reference volume's
**corner origin**. Particle sampling replaces grid sampling. STAR, ndjson, copick
and Portal picks are also supported; voxel-valued picks need the appropriate
`--particles-voxel` spacing. Do not reinterpret centered coordinates as corner-origin
text. Small particle sets may have no held-out subset.

## Movie motion

For a `.mcaln` file produced using the AreTomo3 motion-I/O patch:

```bash
cets-nonrigid to-cets mcaln input/aretomo3/movie_001.mcaln \
  -o exchange/movie_001-aretomo.cets.json --grid 11x11
cets-nonrigid validate exchange/movie_001-aretomo.cets.json
```

The file supplies image geometry, frame integration and reference-frame information.
The result is a separate `MovieAlignment` with global translations and local motion.

Continue with [CETS → Warp](cets-to-warp.md), [CETS → RELION](cets-to-relion.md),
or [CETS → AreTomo3](cets-to-aretomo3.md). Use new output names for every run.
`validate` checks structure and integrity; target fitting measures approximation
error. Missing held-out observations mean `not_evaluated`, not a passed fit.
