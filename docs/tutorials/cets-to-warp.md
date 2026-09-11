# Tutorial: CETS → Warp

Fit tilt-series CETS observations to Warp XML, optionally fitting an available 3D
field, or fit movie observations to Warp grids. See
[Warp conventions](../pipelines/warp.md) for the native evaluation order.

## Validate the input

Activate the [package environment](../../README.md#development-installation).
Use the dose-aware bundle from [AreTomo3 → CETS](aretomo3-to-cets.md):

```bash
cets-nonrigid validate exchange/TS_01-aretomo.cets.json
cets-nonrigid inspect exchange/TS_01-aretomo.cets.json
```

JSON and Zarr supply geometry, G and observations G+r. Template-free Warp synthesis
needs known, nonconstant accumulated dose. Import genuine acquisition metadata;
equal-dose placeholders cannot define the normalized-dose coordinate.

## Fit image movement

```bash
cets-nonrigid from-cets warp exchange/TS_01-aretomo.cets.json \
  -o output/TS_01.xml --movement-grid 4x4 \
  --report-bundle exchange/TS_01-warp-fit.cets.json
cets-nonrigid report exchange/TS_01-warp-fit.cets.json
```

The default `--global-mode fit` derives target globals from CETS. A 4 × 4 movement
grid per tilt is then fitted at target premovement coordinates. This grid is the
target model, not source sampling density. The output XML contains alignment and
available CTF metadata; it is not a complete image project.

Review global/local held-out errors in pixels, coverage, status and available depth
comparisons. `--report-bundle` creates a new source bundle with diagnostics and a
new payload; it does not replace the source alignment with fitted Warp globals.

## Fit available volume displacement

For a [Warp-origin bundle](warp-to-cets.md) with present 3D displacement on active
training and held-out samples:

```bash
cets-nonrigid from-cets warp exchange/TS_01-warp.cets.json \
  -o output/TS_01-volume.xml \
  --volume-warp-grid 3x3x2xT --movement-grid 4x4 \
  --report-bundle exchange/TS_01-volume-fit.cets.json
cets-nonrigid report exchange/TS_01-volume-fit.cets.json
```

Volume-grid dimensions are X/Y/Z/dose-time; `T` uses the target tilt count for the
last axis. The 3D fit precedes the remaining image correction, avoiding double
counting. AreTomo3 patch-only observations cannot supply this independent 3D
channel. Choose a grid supported by your samples and review 3D/CTF-depth metrics;
image agreement alone does not imply depth agreement.

## Prepare a project with images

Save `warp-project.json`:

```json
{
  "project": {
    "tilt_stack": "input/TS_01.mrc"
  }
}
```

The unaligned, motion-corrected stack must have the section indices recorded in
CETS, including excluded rows. Config paths resolve from the working directory.

```bash
cets-nonrigid from-cets warp exchange/TS_01-aretomo.cets.json \
  -o output/warp-project --project --movement-grid 4x4 \
  --config warp-project.json \
  --report-bundle exchange/TS_01-warp-project-fit.cets.json
```

For series ID `TS_01`, the project contains:

```text
output/warp-project/
    warp_tiltseries.settings
    warp_tiltseries/TS_01.xml
    tomostar/TS_01.tomostar
    frames/...                    materialized images
```

Review `project_gates`, `project_ready` and the printed `command_hint`. Reconstruction
is not run. Optional `--template-xml` requires compatible geometry and checked row
identity; `--global-mode template` also requires matching rigid globals.

## Movie motion

From a RELION movie bundle:

```bash
cets-nonrigid from-cets warp-movie exchange/movie_001-relion.cets.json \
  -o output/movie_001.xml --local-grid 3x3x4 \
  --report-bundle exchange/movie_001-warp-fit.cets.json
cets-nonrigid report exchange/movie_001-warp-fit.cets.json
```

`--local-grid` is X/Y/time. Global drift goes to temporal GridMovement and local
motion to GridLocal; fitting may redistribute their constant part using the target
gauge. Compare total motion and held-out errors in Å. Output is alignment XML, not
a raw movie or corrected image. Reimport with the correct dimensions, spacing,
frame count and runtime FractionFrames. Every output must be new.
