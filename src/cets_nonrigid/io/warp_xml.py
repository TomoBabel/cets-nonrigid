"""Strict Warp tilt-series XML loading.

warpylib's ``load_meta`` swallows every exception and silently leaves fields
at their defaults, so a malformed or old-format XML yields a silently wrong
model (e.g. zero image/volume dimensions). This adapter validates the raw XML
FIRST, then loads through warpylib and cross-checks the result.

Only the modern format is supported: the root element must carry
``ImageDimensionsAngstrom`` and ``VolumeDimensionsAngstrom``. Older Warp XML
formats are rejected (project decision, 2026-08-28).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from lxml import etree
from warpylib import TiltSeries

from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

_REQUIRED_ATTRS = ("ImageDimensionsAngstrom", "VolumeDimensionsAngstrom")
_REQUIRED_CHILDREN = ("Angles", "Dose", "UseTilt", "AxisAngle", "AxisOffsetX", "AxisOffsetY")


@dataclass
class DimsOverride:
    """Load-time replacement for the root attributes a re-saved Warp XML zeroes
    (``ImageDimensionsAngstrom="0, 0"``): nothing on disk is edited."""

    image_a: tuple | None = None
    volume_a: tuple | None = None
    level_angle_x: float | None = None
    level_angle_y: float | None = None


@dataclass
class WarpSeries:
    model: WarpTiltSeriesModel
    ts: TiltSeries
    xml_bytes: bytes
    n_tilts: int


def _parse_vector(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",")]


def load_warp_tiltseries(xml_path: str | Path, *, dims_override: DimsOverride | None = None) -> WarpSeries:
    """Load and validate a modern Warp tilt-series XML. ``dims_override``
    supplies the image/volume dimensions (and level angles) for XMLs whose
    root attributes are zero (Warp re-saves them that way) without editing
    the file."""
    xml_path = Path(xml_path)
    xml_bytes = xml_path.read_bytes()

    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"{xml_path}: not valid XML: {e}") from e

    if root.tag != "TiltSeries":
        raise ValueError(f"{xml_path}: root element is <{root.tag}>, expected <TiltSeries>")

    for attr in _REQUIRED_ATTRS:
        if root.get(attr) is None:
            raise ValueError(
                f"{xml_path}: missing root attribute {attr!r} - this looks like an old "
                "Warp XML format, which cets_nonrigid does not support; re-export the series "
                "with a current WarpTools"
            )

    img_a = _parse_vector(root.get("ImageDimensionsAngstrom"))
    vol_a = _parse_vector(root.get("VolumeDimensionsAngstrom"))
    if dims_override is not None:
        if dims_override.image_a is not None:
            img_a = [float(v) for v in dims_override.image_a]
        if dims_override.volume_a is not None:
            vol_a = [float(v) for v in dims_override.volume_a]
    if len(img_a) != 2 or min(img_a) <= 0:
        raise ValueError(f"{xml_path}: bad ImageDimensionsAngstrom {img_a}")
    if len(vol_a) != 3 or min(vol_a) <= 0:
        raise ValueError(f"{xml_path}: bad VolumeDimensionsAngstrom {vol_a}")

    for child in _REQUIRED_CHILDREN:
        el = root.find(child)
        if el is None or not (el.text or "").strip():
            raise ValueError(f"{xml_path}: missing or empty <{child}> element")
    n_tilts = len(root.find("Angles").text.strip().split("\n"))

    mag = root.get("MagnificationCorrection")
    if mag is not None:
        m = _parse_vector(mag)
        if abs(m[0] - m[3]) > 1e-3 or abs(m[1]) > 1e-3 or abs(m[2]) > 1e-3:
            # Not part of the position model, but worth surfacing.
            import warnings

            warnings.warn(
                f"{xml_path}: MagnificationCorrection {m} is non-identity; it does not "
                "enter the position model and is NOT converted",
                stacklevel=2,
            )

    ts = TiltSeries(path=str(xml_path))
    if dims_override is not None:
        ts.image_dimensions_physical = torch.tensor(img_a, dtype=torch.float32)
        ts.volume_dimensions_physical = torch.tensor(vol_a, dtype=torch.float32)
        if dims_override.level_angle_x is not None:
            ts.level_angle_x = float(dims_override.level_angle_x)
        if dims_override.level_angle_y is not None:
            ts.level_angle_y = float(dims_override.level_angle_y)

    # Cross-check the silent warpylib loader against our own parse.
    if ts.n_tilts != n_tilts:
        raise ValueError(
            f"{xml_path}: warpylib loaded {ts.n_tilts} tilts but the XML has {n_tilts} - "
            "the metadata load failed silently"
        )
    movie_path_el = root.find("MoviePath")
    if movie_path_el is not None and (movie_path_el.text or "").strip():
        # Parse MoviePath from the raw XML ourselves and override warpylib's
        # result: warpylib drops blank entries, which destroys the per-tilt row
        # correspondence for mixed populated/blank (dark) rows. Blank rows are
        # data; empty lines at the ends are element-formatting artifacts and are
        # trimmed only while the count exceeds the tilt count.
        entries = [x.strip() for x in movie_path_el.text.split("\n")]
        while len(entries) > n_tilts and entries[0] == "":
            entries.pop(0)
        while len(entries) > n_tilts and entries[-1] == "":
            entries.pop()
        if len(entries) != n_tilts:
            raise ValueError(
                f"{xml_path}: MoviePath has {len(entries)} entries for "
                f"{n_tilts} tilts - the per-tilt row correspondence is broken"
            )
        ts.tilt_movie_paths = entries
    if not torch.allclose(
        ts.image_dimensions_physical, torch.tensor(img_a, dtype=torch.float32)
    ) or not torch.allclose(
        ts.volume_dimensions_physical, torch.tensor(vol_a, dtype=torch.float32)
    ):
        raise ValueError(f"{xml_path}: warpylib dimension load failed silently")

    check_dose_policy(ts, context=str(xml_path))

    return WarpSeries(
        model=WarpTiltSeriesModel(ts),
        ts=ts,
        xml_bytes=xml_bytes,
        n_tilts=n_tilts,
    )


def check_dose_policy(ts, *, context: str) -> None:
    """Equal-dose policy (one rule for loading, fitting and writing): REFUSE.

    Warp normalizes the volume-warp dose axis with ``DoseStep = 1f / (MaxDose -
    MinDose)`` (``TiltSeries.cs:411``): for an equal-dose series that is +Inf,
    the temporal coordinate ``0 * Inf`` is NaN and the native quadrilinear
    evaluator (``NativeAcceleration/src/Einspline.cpp:253-268``) returns NaN for
    EVERY grid, including a 1x1x1x1 one. VERIFIED 2026-09-09 (a) on the real
    managed ``GetPositionInAllTilts`` through ``tools/warpgolden``: all 3075
    positions NaN for both a 1x1x1x1 and a 3x3x2x4 grid; (b) with WarpTools
    2.0.0 ``ts_reconstruct`` on a real project whose only change was equal
    doses: every attempt dies in ``GetCTFsForOneParticle`` (``TiltSeries.cs:908``,
    NaN defocus cast to ``decimal`` -> OverflowException) while the unmodified
    project reconstructs in 23 s. warpylib would substitute ``dose_step = 0``
    (``positions.py:217-218``), but a file Warp cannot evaluate has no valid
    interpretation, so cets_nonrigid refuses it at load and at write with a nudge
    to supply the dose. Dose is Warp's time axis; only synthesized metadata
    ever produces equal doses.
    """
    from cets_nonrigid.models.warp_ts import EQUAL_DOSE_NUDGE, dose_range

    if ts.n_tilts == 0 or dose_range(ts) > 0:
        return
    raise ValueError(f"{context}: {EQUAL_DOSE_NUDGE}")


# ---------------------------------------------------------------------------
# Template-preserving writer
# ---------------------------------------------------------------------------

_PER_TILT_FMT = "{:.6f}"


def _set_per_tilt(root: etree._Element, name: str, values) -> None:
    el = root.find(name)
    if el is None:
        el = etree.SubElement(root, name)
    el.text = "\n".join(_PER_TILT_FMT.format(float(v)) for v in values)


def _replace_grid(root: etree._Element, name: str, grid) -> None:
    old = root.find(name)
    el = etree.Element(name)
    grid.save_to_xml(el)
    if old is not None:
        root.replace(old, el)
    else:
        root.append(el)


_CTF_PARAM_FMT = {
    "Defocus": "{:.9g}",
    "DefocusDelta": "{:.9g}",
    "DefocusAngle": "{:.9g}",
    "PhaseShift": "{:.9g}",
    "Voltage": "{:.9g}",
    "Cs": "{:.9g}",
    "Amplitude": "{:.9g}",
}


def _update_ctf_params(root: etree._Element, ctf) -> None:
    """Update only the owned Param values inside the template's <CTF> element
    (creating element/params as needed); template-only params (BeamTilt,
    Zernikes, ...) are preserved byte-for-byte."""
    el = root.find("CTF")
    if el is None:
        el = etree.SubElement(root, "CTF")
    values = {
        "Defocus": ctf.defocus,
        "DefocusDelta": ctf.defocus_delta,
        "DefocusAngle": ctf.defocus_angle,
        "PhaseShift": ctf.phase_shift,
        "Voltage": ctf.voltage,
        "Cs": ctf.cs,
        "Amplitude": ctf.amplitude,
    }
    for name, value in values.items():
        param = el.find(f"Param[@Name='{name}']")
        if param is None:
            param = etree.SubElement(el, "Param")
            param.set("Name", name)
        param.set("Value", _CTF_PARAM_FMT[name].format(float(value)))


def write_alignment_into_template(
    template_bytes: bytes,
    ts,
    out_path: str | Path,
    *,
    with_ctf: bool = False,
) -> None:
    """Write ``ts``'s ALIGNMENT state into a copy of the template XML.

    Replaces only alignment-owned content — per-tilt Angles/UseTilt/AxisAngle/
    AxisOffsetX/Y, the movement and volume-warp grids, and the LevelAngle
    attributes — and leaves every other node (CTF fits, options, dose, paths)
    byte-for-byte as in the template. With ``with_ctf=True`` the four per-tilt
    CTF grids (GridCTF/DefocusDelta/DefocusAngle/Phase) and the owned <CTF>
    Param values are additionally written from ``ts``. Never mutates the
    input; refuses to overwrite an existing output.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")

    root = etree.fromstring(template_bytes)

    root.set("LevelAngleX", f"{float(ts.level_angle_x):.9g}")
    root.set("LevelAngleY", f"{float(ts.level_angle_y):.9g}")

    _set_per_tilt(root, "Angles", ts.angles.tolist())
    use = root.find("UseTilt")
    if use is None:
        use = etree.SubElement(root, "UseTilt")
    use.text = "\n".join("True" if bool(u) else "False" for u in ts.use_tilt)
    _set_per_tilt(root, "AxisAngle", ts.tilt_axis_angles.tolist())
    _set_per_tilt(root, "AxisOffsetX", ts.tilt_axis_offset_x.tolist())
    _set_per_tilt(root, "AxisOffsetY", ts.tilt_axis_offset_y.tolist())

    _replace_grid(root, "GridMovementX", ts.grid_movement_x)
    _replace_grid(root, "GridMovementY", ts.grid_movement_y)
    _replace_grid(root, "GridVolumeWarpX", ts.grid_volume_warp_x)
    _replace_grid(root, "GridVolumeWarpY", ts.grid_volume_warp_y)
    _replace_grid(root, "GridVolumeWarpZ", ts.grid_volume_warp_z)

    if with_ctf:
        _replace_grid(root, "GridCTF", ts.grid_ctf_defocus)
        _replace_grid(root, "GridCTFDefocusDelta", ts.grid_ctf_defocus_delta)
        _replace_grid(root, "GridCTFDefocusAngle", ts.grid_ctf_defocus_angle)
        _replace_grid(root, "GridCTFPhase", ts.grid_ctf_phase)
        _update_ctf_params(root, ts.ctf)

    payload = etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
    _validate_serialized_dose(payload, ts, context=str(out_path))
    tmp = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _validate_serialized_dose(payload: bytes, ts, *, context: str) -> None:
    """The template's <Dose> is what the output carries (``ts.dose`` is NOT
    written); re-parse it from the serialized bytes and apply the equal-dose
    policy. Runs before publication so a rejection never leaves an invalid file
    behind."""
    from cets_nonrigid.models.warp_ts import EQUAL_DOSE_NUDGE

    root = etree.fromstring(payload)
    dose_el = root.find("Dose")
    if dose_el is None or not (dose_el.text or "").strip():
        raise ValueError(f"{context}: the serialized XML carries no <Dose> column; {EQUAL_DOSE_NUDGE}")
    doses = [float(v) for v in dose_el.text.split()]  # newline-separated per-tilt list
    if len(doses) > 1 and max(doses) - min(doses) <= 0:
        raise ValueError(f"{context}: refusing to write - the serialized XML's <Dose> (the template's) {EQUAL_DOSE_NUDGE}")


def write_movie_alignment_into_template(
    template_bytes: bytes,
    movie,
    out_path: str | Path,
) -> None:
    """Write a Movie's MOTION state into a copy of a movie-XML template.

    Replaces GridMovementX/Y and GridLocalMovementX/Y, removes any
    PyramidShift nodes (cets_nonrigid writes none), and preserves every other node
    (CTF fits, options). Refuses to overwrite an existing output.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")

    root = etree.fromstring(template_bytes)
    if root.tag != "Movie":
        raise ValueError(f"movie template root is <{root.tag}>, expected <Movie>")

    _replace_grid(root, "GridMovementX", movie.grid_movement_x)
    _replace_grid(root, "GridMovementY", movie.grid_movement_y)
    _replace_grid(root, "GridLocalMovementX", movie.grid_local_x)
    _replace_grid(root, "GridLocalMovementY", movie.grid_local_y)
    for name in ("PyramidShiftX", "PyramidShiftY"):
        for el in root.findall(name):
            root.remove(el)

    out_path.write_bytes(
        etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)
    )
