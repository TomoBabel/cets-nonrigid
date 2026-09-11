# Tutorial: Warp → CETS

Import tilt-series XML as CETS globals, projected residuals and available 3D/depth
channels. See [Warp conventions](../pipelines/warp.md) for the evaluation chain.

## Inputs

Activate the [package environment](../../README.md#development-installation).
This example uses a native project and 1.54 Å tilt-image pixels; replace paths
and values with yours:

```text
input/warp/
    warp_tiltseries.settings
    warp_tiltseries/TS_01.xml
    tomostar/TS_01.tomostar
    frames/...
```

The XML must have resolvable image/reference-volume dimensions and genuine doses.
Settings and adjacent project files aid discovery. Pixel size alone cannot supply
a missing box. The required `warpylib` backend comes from the package environment.

## Import and check

```bash
cets-nonrigid to-cets warp input/warp/warp_tiltseries/TS_01.xml \
  -o exchange/TS_01-warp.cets.json \
  --settings input/warp/warp_tiltseries.settings \
  --pix 1.54 --grid 15x15x5
cets-nonrigid validate exchange/TS_01-warp.cets.json
cets-nonrigid inspect exchange/TS_01-warp.cets.json
```

This creates `exchange/TS_01-warp.cets.json` and
`exchange/TS_01-warp.nonrigid.zarr`. Keep them together when moving the bundle.
G contains native rigid parameters. All tilt-series movement and volume-warp grids
remain deformation, including constant grids. The residual already includes the
projected 3D displacement: image observations are **G+r**, without adding it again.

Inspect image identities, active/excluded rows, sample counts and available
channels. `--grid` specifies reference-volume samples; held-out points are separate.
Increasing density requires a new import from the native model. All-equal doses
are refused because Warp's normalized-dose coordinate is undefined.

## Include particles for RELION

Sample the intended particles now for later trajectory export:

```bash
cets-nonrigid to-cets warp input/warp/warp_tiltseries/TS_01.xml \
  -o exchange/TS_01-warp-particles.cets.json \
  --settings input/warp/warp_tiltseries.settings --pix 1.54 \
  --particles input/TS_01-particles.txt
cets-nonrigid validate exchange/TS_01-warp-particles.cets.json
```

The text file contains one `x y z` triple per line in Å from the reference volume's
**corner origin**. Particle sampling replaces the grid and preserves stable point
identities. A grid-only bundle cannot later supply trajectories at new locations.
Follow [CETS → RELION](cets-to-relion.md) to export.

## Movie motion

Movie XML needs explicit runtime context. This example assumes 4096 × 4096 pixels
at 1.54 Å, 40 sampled frames and FractionFrames = 1; use your actual values:

```bash
cets-nonrigid to-cets warp-movie input/warp/movie_001.xml \
  -o exchange/movie_001-warp.cets.json \
  --pix 1.54 --image-size 4096x4096 --n-frames 40 \
  --fraction-frames 1 --grid 11x11
cets-nonrigid validate exchange/movie_001-warp.cets.json
cets-nonrigid inspect exchange/movie_001-warp.cets.json
```

Temporal-only GridMovement becomes the per-frame global translation; GridLocal
and pyramids become residuals. A spatially varying movie GridMovement fails the
baseline gate. This classification differs from tilt-series GridMovement.

All outputs must be new. See [CETS → AreTomo3](cets-to-aretomo3.md) or
[CETS → Warp](cets-to-warp.md) for fits and diagnostics. Structural validation alone
does not establish target approximation accuracy.
