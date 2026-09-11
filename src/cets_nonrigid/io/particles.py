"""Particle-position input for the to-RELION pipelines.

Accepted sources (all yield canonical corner-origin Angstrom positions +
STAR-safe unique names):

  * RELION coordinate/particle star — rlnCenteredCoordinate*Angst (RELION 5)
    or legacy rlnCoordinateX/Y/Z px, with origins/subtomogram angles applied
    per the verified coordinate contract; per-particle optics-group pixel
    sizes must match the tomogram's (v1 rejects mismatches — coordinates and
    trajectories would otherwise need separate scales).
  * plain text — one ``x y z`` triple per line, Angstrom, corner origin.
  * copick — locations are corner-origin, zero-indexed, right-handed physical
    Angstrom (https://copick.github.io/copick/geometry/): canonical directly.
    Requires the ``copick`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cets_nonrigid.models.relion_ts import effective_positions_a

_F64 = torch.float64


def _validate(positions: torch.Tensor, tomo_dims_px, pixel_size_a: float, what: str) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(f"{what}: expected a non-empty (P, 3) position array")
    if not torch.isfinite(positions).all():
        raise ValueError(f"{what}: non-finite particle positions")
    vol_a = torch.tensor(tomo_dims_px, dtype=_F64) * pixel_size_a
    bad = ((positions < 0) | (positions > vol_a)).any(dim=1)
    if bad.any():
        raise ValueError(
            f"{what}: {int(bad.sum())} particle position(s) outside the tomogram "
            f"volume {vol_a.tolist()} A"
        )


def load_star_particles(
    path: str | Path,
    *,
    tomo_name: str,
    tomo_dims_px,
    pixel_size_a: float,
    pixel_size_rtol: float = 1e-3,
) -> tuple:
    from cets_nonrigid.io.relion_star import read_particles_star

    read = read_particles_star(path)
    if read.tomo_names and any(t not in ("", tomo_name) for t in read.tomo_names):
        others = sorted({t for t in read.tomo_names if t not in ("", tomo_name)})
        raise ValueError(
            f"{path}: contains particles of other tomograms {others}; select --tomo-name rows first"
        )
    # v1 pixel-size consistency: every selected optics group must match the tomogram
    for grp, pix in read.optics_pixel_size_a.items():
        if abs(pix - pixel_size_a) > pixel_size_rtol * pixel_size_a:
            raise ValueError(
                f"{path}: optics group {grp} rlnTomoTiltSeriesPixelSize {pix} != tomogram "
                f"pixel size {pixel_size_a} — unequal values would need separate coordinate "
                "and trajectory scales (unsupported in v1)"
            )
    positions = effective_positions_a(
        pixel_size_a=pixel_size_a,
        tomo_dims_px=tuple(tomo_dims_px),
        centered_coords_a=read.centered_coords_a,
        legacy_coords_px=read.legacy_coords_px,
        origins_a=read.origins_a,
        subtomo_angles_deg=read.subtomo_angles_deg,
    )
    _validate(positions, tomo_dims_px, pixel_size_a, str(path))
    return positions, list(read.particle_names)


def load_text_particles(
    path: str | Path,
    *,
    tomo_name: str,
    tomo_dims_px,
    pixel_size_a: float,
) -> tuple:
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    positions = torch.tensor(arr, dtype=_F64)
    _validate(positions, tomo_dims_px, pixel_size_a, str(path))
    names = [f"{tomo_name}/{i + 1}" for i in range(positions.shape[0])]
    return positions, names


def voxel_size_from_tomogram(tomogram_path: str | Path, *, pixel_size_a: float, raw_x_px: int) -> float:
    """Voxel size implied by the raw field: ``pix * raw_x / nx_header`` (the
    header voxel of an AreTomo3/portal tomogram can be rounded, e.g. 4.99 vs
    5.006 A); warns when the header disagrees by more than 1e-3 relative."""
    import warnings

    import mrcfile

    with mrcfile.open(str(tomogram_path), header_only=True, permissive=True) as m:
        nx = int(m.header.nx)
        header_voxel = float(m.voxel_size.x)
    implied = float(pixel_size_a) * int(raw_x_px) / nx
    if header_voxel > 0 and abs(header_voxel - implied) > 1e-3 * implied:
        warnings.warn(
            f"{tomogram_path}: header voxel {header_voxel:.4f} A differs from the raw-field-implied "
            f"{implied:.4f} A; using the implied value",
            stacklevel=2,
        )
    return implied


def load_ndjson_particles(
    path: str | Path,
    *,
    tomo_name: str,
    tomo_dims_px,
    pixel_size_a: float,
    unit: str = "voxel",
    voxel_size_a: float | None = None,
) -> tuple:
    """cryoET Data Portal point annotations: one JSON object per line with
    ``location: {x, y, z}`` in tomogram voxel units (index convention)."""
    import json

    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        loc = obj.get("location", obj)
        rows.append([float(loc["x"]), float(loc["y"]), float(loc["z"])])
    if not rows:
        raise ValueError(f"{path}: no points")
    positions = torch.tensor(rows, dtype=_F64)
    if unit == "voxel":
        if voxel_size_a is None:
            raise ValueError("voxel-unit picks need voxel_size_a (--particles-voxel-size or --particles-tomogram-dir)")
        positions = positions * float(voxel_size_a)
    elif unit != "angstrom":
        raise ValueError(f"unknown particle unit {unit!r}")
    _validate(positions, tomo_dims_px, pixel_size_a, str(path))
    names = [f"{tomo_name}/{i + 1}" for i in range(positions.shape[0])]
    return positions, names


def load_particles(
    path: str | Path,
    *,
    tomo_name: str,
    tomo_dims_px,
    pixel_size_a: float,
    unit: str = "angstrom",
    voxel_size_a: float | None = None,
) -> tuple:
    """Dispatch by extension: ``.star`` (RELION), ``.ndjson`` (portal points,
    voxel units unless told otherwise), anything else = ``x y z`` text in
    ``unit`` (Angstrom corner origin, or voxel indices times ``voxel_size_a``)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".star":
        return load_star_particles(path, tomo_name=tomo_name, tomo_dims_px=tomo_dims_px, pixel_size_a=pixel_size_a)
    if suffix == ".ndjson":
        return load_ndjson_particles(
            path, tomo_name=tomo_name, tomo_dims_px=tomo_dims_px, pixel_size_a=pixel_size_a,
            unit="voxel" if unit == "angstrom" and voxel_size_a is not None else unit,
            voxel_size_a=voxel_size_a,
        )
    if unit == "voxel":
        if voxel_size_a is None:
            raise ValueError("voxel-unit picks need voxel_size_a (--particles-voxel-size or --particles-tomogram-dir)")
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
        positions = torch.tensor(arr, dtype=_F64) * float(voxel_size_a)
        _validate(positions, tomo_dims_px, pixel_size_a, str(path))
        return positions, [f"{tomo_name}/{i + 1}" for i in range(positions.shape[0])]
    return load_text_particles(path, tomo_name=tomo_name, tomo_dims_px=tomo_dims_px, pixel_size_a=pixel_size_a)


def load_copick_uri_particles(
    config: str | Path,
    uri: str,
    *,
    run_name: str,
    tomo_name: str,
    tomo_dims_px,
    pixel_size_a: float,
) -> tuple:
    """Picks addressed by the copick URI ``object_name:user_id/session_id``
    (glob/regex patterns allowed) within the run ``run_name``; the URI must
    resolve to exactly one pick set."""
    import copick
    from copick.util.uri import resolve_copick_objects

    root = copick.from_file(str(config))
    if root.get_run(run_name) is None:
        raise ValueError(f"copick run {run_name!r} not found in {config}")
    objs = resolve_copick_objects(uri, root, "picks", run_name=run_name)
    if not objs:
        raise ValueError(f"no picks match {uri!r} in copick run {run_name!r}")
    if len(objs) > 1:
        who = [(p.pickable_object_name, p.user_id, p.session_id) for p in objs]
        raise ValueError(f"picks URI {uri!r} is ambiguous in run {run_name!r}: {who}")
    points = objs[0].points
    positions = torch.tensor([[p.location.x, p.location.y, p.location.z] for p in points], dtype=_F64)
    _validate(positions, tomo_dims_px, pixel_size_a, f"copick:{run_name}/{uri}")
    names = [f"{tomo_name}/{i + 1}" for i in range(positions.shape[0])]
    return positions, names
