"""Template-free Warp model synthesis (shared factory, Workstream T).

Builds a valid warpylib ``TiltSeries``/``Movie`` from explicit metadata when no
template XML is available, serializes it through warpylib's own ``save_meta``,
and strict-reloads the result before handing it back — so a synthesized
"template" behaves exactly like a loaded one downstream (the existing
template-preserving writers then emit the final alignment into it).

Contracts (plan rev. 5):
- Physical geometry is never guessed — callers must supply pixel size and the
  missing dimensions.
- Generated CTF is a *placeholder with every physically known field
  populated*: ``CTF.PixelSize`` is always set from the known pixel size
  (warpylib's default of 1.0 A is never acceptable when the truth is known),
  voltage/Cs/amplitude when known, and missing per-tilt CTF becomes 1x1xT
  grids filled with the scalar placeholder defocus so per-tilt accessors
  behave consistently.
- Serialization is atomic: write to a sibling temporary path, strict-load it,
  then publish with ``os.replace``; refuses to overwrite.
- Defaults are loud: the returned report distinguishes UNAVAILABLE
  EXPERIMENTAL metadata (dose, movie paths, inversion, CTF parameters,
  processing history) from NEUTRAL INITIALIZATION inherent to serialization
  (plane normal, FOV fraction, B-factor/weight, identity magnification, zero
  deformation grids).
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from warpylib import CubicGrid, LinearGrid4D, TiltSeries
from warpylib.movie import Movie

from cets_nonrigid.ctf import TiltCtf, tiltctf_to_warp

#: Neutral serialization state, not missing experimental data.
NEUTRAL_INITIALIZED = (
    "PlaneNormal",
    "FOVFraction",
    "Bfactor/Weight",
    "MagnificationCorrection (identity)",
    "zero movement/deformation grids",
)

#: Warp processing-option history cannot be recreated from any source.
_ALWAYS_UNAVAILABLE = ("Warp processing-option history",)


@dataclass
class SynthReport:
    """What a synthesized model defaulted, split by kind (plan contract)."""

    template_source: str = "generated"
    defaulted_experimental: list = field(default_factory=list)
    neutral_initialized: list = field(default_factory=lambda: list(NEUTRAL_INITIALIZED))

    def warn_once(self, context: str) -> None:
        warnings.warn(
            f"{context}: no template given - generated Warp XML. "
            f"Unavailable experimental defaults: {', '.join(self.defaulted_experimental)}. "
            f"Neutral initialization: {', '.join(self.neutral_initialized)}.",
            stacklevel=3,
        )


def _placeholder_ctf_grids(ts: TiltSeries) -> None:
    """1x1xT CTF grids filled with the scalar placeholder values — a single
    default grid node gives inconsistent per-tilt accessor behavior."""
    t = ts.n_tilts
    ts.grid_ctf_defocus = CubicGrid(
        (1, 1, t), values=torch.full((t,), float(ts.ctf.defocus), dtype=torch.float32)
    )
    ts.grid_ctf_defocus_delta = CubicGrid((1, 1, t), values=torch.zeros(t, dtype=torch.float32))
    ts.grid_ctf_defocus_angle = CubicGrid((1, 1, t), values=torch.zeros(t, dtype=torch.float32))
    ts.grid_ctf_phase = CubicGrid((1, 1, t), values=torch.zeros(t, dtype=torch.float32))


def _apply_known_ctf_scalars(
    ctf,
    pixel_size_a: float,
    voltage_kv: float | None,
    cs_mm: float | None,
    amplitude_contrast: float | None,
    report: SynthReport,
) -> None:
    ctf.pixel_size = float(pixel_size_a)
    if voltage_kv is not None:
        ctf.voltage = float(voltage_kv)
    if cs_mm is not None:
        ctf.cs = float(cs_mm)
    if amplitude_contrast is not None:
        ctf.amplitude = float(amplitude_contrast)
    unknown = [
        name
        for name, value in (
            ("voltage", voltage_kv),
            ("Cs", cs_mm),
            ("amplitude contrast", amplitude_contrast),
        )
        if value is None
    ]
    if unknown:
        report.defaulted_experimental.append(f"CTF scalars ({', '.join(unknown)})")


def atomic_write_validated(write_fn, out_path: str | Path, validate_fn) -> None:
    """Serialize via ``write_fn(tmp_path)``, run ``validate_fn(tmp_path)``,
    then atomically publish. A failed validation never leaves output behind;
    refuses to overwrite an existing destination."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")
    # keep the real extension (loaders derive sibling paths from it); the
    # uniqueness lives in the prefix
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=".tmp-", suffix=out_path.suffix)
    os.close(fd)
    os.unlink(tmp)  # writers refuse existing paths; keep only the reserved name
    try:
        write_fn(tmp)
        validate_fn(tmp)
        os.replace(tmp, out_path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def synthesize_tilt_series(
    *,
    angles_deg,
    use_tilt,
    axis_angles_deg,
    axis_offset_x_a,
    axis_offset_y_a,
    image_dims_a,
    volume_dims_a,
    pixel_size_a: float,
    dose=None,
    movie_paths=None,
    angles_inverted: bool = False,
    angles_inverted_known: bool = False,
    level_angle_x: float = 0.0,
    level_angle_y: float = 0.0,
    voltage_kv: float | None = None,
    cs_mm: float | None = None,
    amplitude_contrast: float | None = None,
    tilt_ctf: TiltCtf | None = None,
) -> tuple[TiltSeries, SynthReport]:
    """Build a fresh, fully populated warpylib ``TiltSeries`` (raw row order,
    dark rows included with ``use_tilt=False``). Geometry is mandatory; every
    default is recorded in the returned :class:`SynthReport`."""
    report = SynthReport()
    report.defaulted_experimental.extend(_ALWAYS_UNAVAILABLE)

    angles = torch.as_tensor(angles_deg, dtype=torch.float32)
    t = int(angles.shape[0])
    if t == 0:
        raise ValueError("cannot synthesize a tilt series with zero tilts")

    def per_tilt(name, values):
        v = torch.as_tensor(values, dtype=torch.float32)
        if v.shape[0] != t:
            raise ValueError(f"{name} has {v.shape[0]} entries for {t} tilts")
        return v

    img = torch.as_tensor(image_dims_a, dtype=torch.float32)
    vol = torch.as_tensor(volume_dims_a, dtype=torch.float32)
    if img.shape != (2,) or float(img.min()) <= 0:
        raise ValueError(f"bad image dims (A): {img.tolist()}")
    if vol.shape != (3,) or float(vol.min()) <= 0:
        raise ValueError(f"bad volume dims (A): {vol.tolist()}")
    if pixel_size_a <= 0:
        raise ValueError(f"bad pixel size: {pixel_size_a}")

    ts = TiltSeries(n_tilts=t)
    ts.angles = angles
    ts.use_tilt = torch.as_tensor(use_tilt, dtype=torch.bool)
    if ts.use_tilt.shape[0] != t:
        raise ValueError(f"use_tilt has {ts.use_tilt.shape[0]} entries for {t} tilts")
    ts.tilt_axis_angles = per_tilt("axis_angles_deg", axis_angles_deg)
    ts.tilt_axis_offset_x = per_tilt("axis_offset_x_a", axis_offset_x_a)
    ts.tilt_axis_offset_y = per_tilt("axis_offset_y_a", axis_offset_y_a)
    ts.image_dimensions_physical = img
    ts.volume_dimensions_physical = vol
    ts.level_angle_x = float(level_angle_x)
    ts.level_angle_y = float(level_angle_y)

    ts.are_angles_inverted = bool(angles_inverted)
    if not angles_inverted_known:
        report.defaulted_experimental.append("AreAnglesInverted=False")

    if dose is None:
        from cets_nonrigid.models.warp_ts import EQUAL_DOSE_NUDGE

        raise ValueError(
            "cannot synthesize a Warp tilt series without a per-tilt dose (no template carries one): "
            f"a zero dose would mean {EQUAL_DOSE_NUDGE}"
        )
    ts.dose = per_tilt("dose", dose)

    if movie_paths is not None:
        paths = [str(p) for p in movie_paths]
        if len(paths) != t:
            raise ValueError(
                f"movie_paths must have exactly one row per raw section (darks included): "
                f"got {len(paths)} for {t} sections"
            )
        if any(not p for p in paths):
            raise ValueError(
                "movie_paths is partially specified - provide a path for every raw "
                "section (darks included) or none at all"
            )
        ts.tilt_movie_paths = paths
    else:
        ts.tilt_movie_paths = [""] * t
        report.defaulted_experimental.append("blank movie paths")

    ts.fov_fraction = torch.ones(t, dtype=torch.float32)

    ts.grid_movement_x = CubicGrid((1, 1, 1))
    ts.grid_movement_y = CubicGrid((1, 1, 1))
    ts.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))

    _apply_known_ctf_scalars(ts.ctf, pixel_size_a, voltage_kv, cs_mm, amplitude_contrast, report)
    if tilt_ctf is not None:
        tiltctf_to_warp(ts, tilt_ctf)
    else:
        _placeholder_ctf_grids(ts)
        report.defaulted_experimental.append("placeholder per-tilt CTF")

    return ts, report


def synthesized_template_series(ts: TiltSeries, context: str):
    """Serialize a synthesized TiltSeries and strict-reload it, returning the
    same ``WarpSeries`` a loaded template would give (so every downstream
    consumer is agnostic to how the template came to be)."""
    from cets_nonrigid.io.warp_xml import load_warp_tiltseries

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "synthesized.xml"
        ts.save_meta(str(tmp))
        try:
            series = load_warp_tiltseries(tmp)
        except ValueError as e:
            raise ValueError(f"{context}: synthesized template failed strict reload: {e}") from e
    return series


def synthesize_movie(
    *,
    pixel_size_a: float,
    voltage_kv: float | None = None,
    cs_mm: float | None = None,
    amplitude_contrast: float | None = None,
    data_path: str | None = None,
) -> tuple[Movie, SynthReport]:
    """Fresh warpylib ``Movie`` with known metadata populated. Motion grids are
    fitted by the caller afterwards; a generated movie XML is usable only
    alongside the matching raw movie and externally supplied runtime metadata
    (dims, frame count, FractionFrames are never stored in the XML)."""
    if pixel_size_a <= 0:
        raise ValueError(f"bad pixel size: {pixel_size_a}")
    report = SynthReport()
    report.defaulted_experimental.extend(_ALWAYS_UNAVAILABLE)
    movie = Movie(path=data_path)
    _apply_known_ctf_scalars(movie.ctf, pixel_size_a, voltage_kv, cs_mm, amplitude_contrast, report)
    report.defaulted_experimental.append("placeholder CTF fit")
    return movie, report


def movie_template_bytes(movie: Movie, context: str) -> bytes:
    """Serialize a synthesized Movie and strict-reload it, returning template
    bytes for ``write_movie_alignment_into_template``."""
    from cets_nonrigid.io.warp_movie_xml import load_warp_movie_strict

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "synthesized.xml"
        movie.save_meta(str(tmp))
        try:
            load_warp_movie_strict(tmp)
        except ValueError as e:
            raise ValueError(f"{context}: synthesized movie failed strict reload: {e}") from e
        return tmp.read_bytes()
