# Tutorial: CETS → RELION 5

Export a CETS tilt-series alignment as a RELION/py2rely project with particle
trajectories. See [RELION conventions](../pipelines/relion.md) for lift and CTF gauges.

## Prepare the inputs

Activate the [package environment](../../README.md#development-installation).
Use the particle-bound bundle from [Warp → CETS](warp-to-cets.md), or an equivalent
[AreTomo3](aretomo3-to-cets.md) or [RELION](relion-to-cets.md) import. The intended
particles must already have been sampled.

```bash
cets-nonrigid validate exchange/TS_01-warp-particles.cets.json
cets-nonrigid inspect exchange/TS_01-warp-particles.cets.json
```

This example requires known per-image pre-exposure, per-tilt CTF, defocus handedness
and voltage/Cs/amplitude contrast in CETS. Supply `input/TS_01.mrc`, an unaligned,
motion-corrected tilt stack with the section indices recorded in CETS.

## Export and check

```bash
cets-nonrigid from-cets relion exchange/TS_01-warp-particles.cets.json \
  -o output/relion-project --tilt-stack input/TS_01.mrc \
  --trajectory-gauge lowest-dose \
  --report-bundle exchange/TS_01-relion-fit.cets.json
cets-nonrigid report exchange/TS_01-relion-fit.cets.json
```

The output is a new project directory:

```text
output/relion-project/
    optimisation_set.star
    tomograms.star
    tilt_series/TS_01.star
    particles.star
    motion.star
    images/TS_01/...             extracted active tilt images
```

STAR-internal paths are project-root relative. Keep this layout together. Conversion
prepares files without launching extraction/reconstruction. If CETS already names
suitable image assets, omit `--tilt-stack`; `--tilt-image-list` also accepts one
image reference per active emitted tilt in emitted row order.

The exporter lifts complete image observations G+r into trajectories, reuniting
training and held-out particles. It does not reconstruct the original deformation
coefficients or infer values at unsampled particles.

## Gauges and diagnostics

`lowest-dose` anchors the static coordinate to the lowest-dose emitted row.
`ctf-optimal` minimizes static-coordinate CTF-depth discrepancy. To compare gauges,
repeat with `--trajectory-gauge ctf-optimal`, a new output directory and a new
report-bundle name.

Review particle/tilt counts, `max_residual_px`, `gauge`, fallback counts and available
CTF-depth deviations. The projected lift is checked against 0.001 pixel; this does
not ensure equal CTF depth. Missing depth/held-out comparisons are not passes.
The report bundle copies source observations with diagnostics; it is not a reimport
of the emitted RELION alignment.

Supply missing **known** metadata using `--voltage` (kV), `--cs` (mm),
`--amp-contrast` (fraction) and `--hand -1` or `--hand 1` (defocus handedness).
Handedness should not be inferred from an unrelated image flip.

## Global-only output

A grid-only bundle can intentionally omit particle deformation:

```bash
cets-nonrigid from-cets relion exchange/TS_01-warp.cets.json \
  -o output/relion-global-project --no-particles \
  --tilt-stack input/TS_01.mrc
```

Dose, handedness, optics and CTF are still needed. For explicitly geometry-only
inspection, `--no-ctf` writes labelled placeholder CTF, not estimated scientific
CTF. To retain deformation from a grid source, reimport its native input at the
intended particle positions.

## Movie motion

From a Warp movie bundle:

```bash
cets-nonrigid from-cets relion-motion exchange/movie_001-warp.cets.json \
  -o output/movie_001.star --movie-name input/movie_001.eer \
  --dose-rate 0.5 --pre-exposure 2.0 --voltage 300 \
  --report-bundle exchange/movie_001-relion-fit.cets.json
cets-nonrigid report exchange/movie_001-relion-fit.cets.json
```

Replace example dose per sampled frame (electrons/Å²), pre-exposure (electrons/Å²),
voltage (kV) and path with genuine values; omit overrides already in CETS. Output
is a per-micrograph motion STAR with globals and fitted local polynomials. The raw
movie is referenced, not copied. Review held-out motion error in Å and frame
identities. Every native output and report bundle must be new.
