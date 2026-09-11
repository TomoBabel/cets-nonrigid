# Tutorial: CETS → AreTomo3

Fit CETS tilt-series observations to `.aln` patches, or movie observations to
`.mcaln`. Coefficients need not match the source: the exchange preserves observables
at samples. See [AreTomo3 conventions](../pipelines/aretomo3.md).

## Validate the input

Activate the [package environment](../../README.md#development-installation).
This example uses the bundle from [Warp → CETS](warp-to-cets.md). Other tilt-series
sources work when their global geometry is representable in AreTomo3.

```bash
cets-nonrigid validate exchange/TS_01-warp.cets.json
cets-nonrigid inspect exchange/TS_01-warp.cets.json
```

Keep the Zarr payload beside the JSON. CETS provides geometry and G; native
snapshots are not needed. The global fit must pass before patches are fitted.
An unsupported X tilt cannot be hidden in the patch field.

## Fit an alignment file

```bash
cets-nonrigid from-cets aretomo3 exchange/TS_01-warp.cets.json \
  -o output/TS_01.aln --patch-grid 4x4 --patch-z lsq \
  --report-bundle exchange/TS_01-aretomo-fit.cets.json
cets-nonrigid report exchange/TS_01-aretomo-fit.cets.json
```

`--patch-grid` sets target X/Y patch counts independently of source sampling.
`--patch-z lsq` selects least-squares patch depth; `zero` selects zero depth.
Patches fit complete observations G+r against the target's projection coordinates.
The command writes `.aln` and applicable companions, and prints metrics.

`--report-bundle` makes a **new source bundle with diagnostics**, including a new
payload. It does not replace the source alignment with the fitted target. Review
`global_rms_px_heldout`, `rms_px_heldout`, coverage and status when available.
`not_evaluated` is not zero error. Choose grid density using support and held-out
error; investigate a failed global gate before changing its tolerance.

## Prepare a complete project

A standalone `.aln` does not contain image pixels. Save `aretomo-project.json`:

```json
{
  "project": {
    "tilt_stack": "input/TS_01.mrc"
  }
}
```

Supply the unaligned, motion-corrected stack whose section indices match CETS,
including excluded sections where present. Config paths resolve from the working
directory.

```bash
cets-nonrigid from-cets aretomo3 exchange/TS_01-warp.cets.json \
  -o output/aretomo-project --project --patch-grid 4x4 \
  --config aretomo-project.json \
  --report-bundle exchange/TS_01-aretomo-project-fit.cets.json
```

For CETS series ID `TS_01`, the output has `TS_01.aln`, `TS_01.mrc`,
`TS_01_TLT.txt` and available `TS_01_CTF.txt`. The writer maps/reorders image sections
for the emitted alignment. Review `project_gates`, `project_ready` and the printed
`command_hint` before reconstruction. Conversion prepares files without running
AreTomo3. CTF correction also needs sufficient CTF/optics metadata.

## Movie motion

From a Warp movie bundle:

```bash
cets-nonrigid from-cets mcaln exchange/movie_001-warp.cets.json \
  -o output/movie_001.mcaln --patch-grid 4x4 \
  --report-bundle exchange/movie_001-mcaln-fit.cets.json
cets-nonrigid report exchange/movie_001-mcaln-fit.cets.json
```

The fit normalizes the reference-frame gauge and reports motion error in Å.
An existing CETS reference identity is used when available; otherwise the default
is the middle aligned frame. `--fm-ref` selects a zero-based aligned frame index.
Account for this gauge when comparing global shifts. The AreTomo3 binary needs
the motion-I/O patch to consume `.mcaln`; this command writes the text format
without invoking that binary. Every native output and report bundle must be new.
