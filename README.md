# cets-nonrigid

Sampled non-rigid alignment exchange between AreTomo3, Warp and RELION 5,
using CETS as the authoritative model for geometry, globals and scientific metadata.
Tilt-series deformation and movie motion are supported in all twelve directions.

This development package depends on the CETS schema's
[feature/nonrigid-alignment branch](https://github.com/TomoBabel/cets-data-models/tree/feature/nonrigid-alignment).
The schema changes are available on that branch; they are not merged into main.
For local schema development, use the independent `cets-data-models-dev` checkout.

## Model

```text
native alignment ──> CETS Alignment / MovieAlignment
                          globals + non_rigid_alignment descriptor
                                        |
                                        v
                              Zarr sampled residuals
                                        |
                              target fit + validation
                                        |
                                        v
                              native files / project
```

`Alignment.non_rigid_alignment` is a first-class core property.
`has_non_rigid_alignment` is derived and is not serialized. A reference volume is
an ordinary `Tomogram`, including one with a null image path.

The format is lossy: **observables at samples, not model identity**, are the
round-trip invariant. The document defines G; the payload stores r = q − G.
The optional 3D displacement is already projected into r and is never added twice.
See the [exchange profile](docs/exchange-profile.md) for frames, channel masks,
row identities, digest rules and native global/deformation classification.

The native pipeline guides cover projection equations, coordinate/shift conventions,
CETS field mappings, target fitting and movie motion:

- [AreTomo3](docs/pipelines/aretomo3.md): tilt-series patches and `.mcaln` motion.
- [Warp](docs/pipelines/warp.md): volume/image grids, CTF depth and movie grids.
- [RELION 5](docs/pipelines/relion.md): projection matrices, particles, trajectories and motion polynomials.

Step-by-step tutorials cover tilt-series and movie conversion:

| Pipeline | Import into CETS | Export from CETS |
|---|---|---|
| AreTomo3 | [AreTomo3 → CETS](docs/tutorials/aretomo3-to-cets.md) | [CETS → AreTomo3](docs/tutorials/cets-to-aretomo3.md) |
| Warp | [Warp → CETS](docs/tutorials/warp-to-cets.md) | [CETS → Warp](docs/tutorials/cets-to-warp.md) |
| RELION 5 | [RELION → CETS](docs/tutorials/relion-to-cets.md) | [CETS → RELION](docs/tutorials/cets-to-relion.md) |

## Development installation

Python ≥3.11 is declared; the validated environment is Linux x86_64 / Python 3.13,
Torch 2.8.0+cu129 and torch-projectors 0.13.0+cu129. `warpylib` is required.
Exact versions and Git revisions are in [environment-lock.json](environment-lock.json).
A CUDA GPU is required for native CUDA validation and Warp reconstruction.

With the two development repositories next to each other:

```bash
python tools/bootstrap_development.py --environment /path/to/new/cets-env --dry-run
python tools/bootstrap_development.py --environment /path/to/new/cets-env
source /path/to/new/cets-env/bin/activate
python tools/check_environment.py
```

The installer refuses an existing environment. Install custom ABI wheels before
warpylib. Normal package installation obtains the schema from the branch named in
`pyproject.toml`; `environment-lock.json` records the exact validated commit because
a branch can advance. The development bootstrap checks that commit and keeps the
local schema editable instead of replacing it with a Git installation.

## Commands

```bash
cets-nonrigid to-cets aretomo3 series.aln -o series.cets.json \
  --pix 1.54 --tomo-size 4096x4096x2000 --mdoc-dir acquisition --dose-per-tilt 3.87
cets-nonrigid from-cets warp series.cets.json -o series.xml --movement-grid 4x4
cets-nonrigid convert warp-to-aretomo3 series.xml -o output.aln --pix 1.54
cets-nonrigid validate series.cets.json
cets-nonrigid inspect series.cets.json
cets-nonrigid report series.cets.json
```

`fit <target>` also accepts `--report-bundle new.cets.json`. Every output must be
new. JSON is published after its validated payload. Relocate JSON and Zarr together.
Each leaf has its own help and at most 25 options. Advanced API options are available
through `--config`, containing `source`, `target` and `project` objects.

Supply particles during `to-cets` for deformation-aware RELION export; a grid
cannot define trajectories at unsampled particles. `--no-particles` supports
rigid-only RELION output. Movie sources/targets are `warp-movie`, `mcaln` and
`relion-motion`. Multiple inputs or source directories perform batch conversion;
RELION optimisation sets expand to their tomograms unless `--tomo-name` selects one.

## Converter API

```python
from cets_nonrigid import api

native = api.load_native("warp", "series.xml", pixel_size_a=1.54)
# An existing converter may supply its own compatible CetsContext and row_map.
samples = api.sample(native, context=native.context)
bundle = api.attach_deformation(native.context, samples)
api.write_bundle(bundle, "series.cets.json")

bundle = api.read_bundle("series.cets.json")
result = api.fit(bundle, "aretomo3", patch_grid=(4, 4))
api.export_native(result, "converted.aln")
```

Core wire types come from `cets_data_model`; tensor containers and fit results
belong to `cets_nonrigid`. Attachment returns a new document. The baseline gate
rejects a Warp alignment whose deformation constants were already folded into G.
Project writers are available through `prepare_project` and `merge_projects`.
Image assets are streamed on publication; fitting/export never reopen native
snapshots. Keep a `FitResult` alive until its assets have been exported.

Core metadata coverage and pending upstream work are tracked in the
[core-schema checklist](docs/core-schema-checklist.md).

## Validation and scope

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 pytest -q
ruff check src tests tools
python -m mypy
(cd ../cets-data-models-dev && make check-generated)
```

The port excludes anisotropic pixels, non-unit Warp SizeRoundingFactors,
GridAngle*, magnification correction, AreTomo2 and legacy arewarpion stores.
GPL-derived validation tooling builds from external sources in scratch space and
is excluded from wheel/sdist. Transferred MIT code retains attribution; newly
written integration code is MPL-2.0. See [source notices](THIRD_PARTY_NOTICES.md).

[Core additions](docs/cets-additions.md) track the local schema and future upstream
reconciliation. Merging the schema into main and changing downstream converters are later steps.
