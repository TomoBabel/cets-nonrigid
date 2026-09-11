"""Discover a series' metadata from a Warp project: the tilt-series XML
(``<CTF>`` scalars, dimension attributes, ``AreAnglesInverted``, MoviePath,
Dose), the project ``.settings`` (pixel size, tomogram box, CTF constants,
data/processing folders) and the frame averages Warp reads."""

from __future__ import annotations

from pathlib import Path

from cets_nonrigid.meta.resolve import Discovered


def find_warp_settings(xml_path: Path) -> Path | None:
    """``*.settings`` next to the processing folder whose ProcessingFolder
    names it (``<root>/warp_tiltseries.settings`` for ``<root>/warp_tiltseries/x.xml``);
    when none matches but the project has exactly one settings file, that one."""
    from cryoet_alignment.io.warp.settings import WarpSettings

    parent = xml_path.parent.parent
    hits, parsed = [], []
    for cand in sorted(parent.glob("*.settings")):
        try:
            s = WarpSettings.from_file(str(cand))
        except Exception:  # noqa: BLE001, S112 - a *.settings that is not a Warp document is simply skipped
            continue
        parsed.append(cand)
        if s.processing_folder and Path(s.processing_folder).name == xml_path.parent.name:
            hits.append(cand)
    if hits:
        return hits[0]
    # a converted XML in a sibling processing folder (warp_tiltseries_converted/)
    # still belongs to the project's one settings file
    return parsed[0] if len(parsed) == 1 else None


def discover_warp_series(
    xml_path: str | Path,
    *,
    settings: str | Path | None = None,
    frames_dir: str | Path | None = None,
    tilt_stack: str | Path | None = None,
    tilt_stack_dir: str | Path | None = None,
    discover_adjacent: bool = True,
) -> Discovered:
    from lxml import etree

    from cets_nonrigid.io.frames import resolve_average_paths
    from cets_nonrigid.meta.aretomo_run import mrc_header

    xml_path = Path(xml_path)
    stem = xml_path.stem
    d = Discovered(series_name=stem, source=str(xml_path))
    root = etree.fromstring(xml_path.read_bytes())
    ctf = {}
    ctf_el = root.find("CTF")
    if ctf_el is not None:
        for param in ctf_el.findall("Param"):
            try:
                ctf[param.get("Name")] = float(param.get("Value"))
            except (TypeError, ValueError):
                continue

    def _vec(attr):
        raw = root.get(attr)
        if not raw:
            return None
        try:
            vals = [float(v) for v in raw.split(",")]
        except ValueError:
            return None
        return vals if all(v > 0 for v in vals) else None

    def _text(tag):
        el = root.find(tag)
        return [x.strip() for x in (el.text or "").strip().split("\n")] if el is not None and (el.text or "").strip() else []

    angles = _text("Angles")
    n = len(angles)
    d.add("n_raw_sections", n, "file", f"{xml_path.name}#Angles")
    if ctf.get("PixelSize", 0) > 0:
        d.add("pixel_size_a", ctf["PixelSize"], "file", f"{xml_path.name}#CTF/PixelSize")
    for key, field in (("Voltage", "voltage_kv"), ("Cs", "cs_mm"), ("Amplitude", "amplitude_contrast")):
        if key in ctf:
            d.add(field, ctf[key], "file", f"{xml_path.name}#CTF/{key}")
    inv = root.get("AreAnglesInverted")
    if inv is not None:
        d.add("angles_inverted", inv.strip().lower() == "true", "file", f"{xml_path.name}#AreAnglesInverted")

    # --- settings ------------------------------------------------------------
    settings_path = Path(settings) if settings else (find_warp_settings(xml_path) if discover_adjacent else None)
    s = None
    if settings_path is not None and settings_path.exists():
        from cryoet_alignment.io.warp.settings import WarpSettings

        s = WarpSettings.from_file(str(settings_path))
        d.aux["settings"] = str(settings_path)
        if s.pixel_size_a:
            d.add("pixel_size_a", s.pixel_size_a, "file", f"{settings_path.name}#Import/PixelSize")
        if s.tomo_dims_px:
            d.add("tomo_dims_px", tuple(s.tomo_dims_px), "file", f"{settings_path.name}#Tomo/Dimensions")
        for key, field in (("voltage_kv", "voltage_kv"), ("cs_mm", "cs_mm"), ("amplitude_contrast", "amplitude_contrast")):
            v = getattr(s, key)
            if v is not None:
                d.add(field, v, "file", f"{settings_path.name}#CTF")
        if s.exposure_per_tilt:
            d.add("dose_per_tilt", s.exposure_per_tilt, "file", f"{settings_path.name}#Import/DosePerAngstromFrame")

    pix = d.first("pixel_size_a")
    img_a, vol_a = _vec("ImageDimensionsAngstrom"), _vec("VolumeDimensionsAngstrom")
    if pix:
        if img_a:
            d.add("image_dims_px", tuple(round(v / pix) for v in img_a), "derived", f"{xml_path.name}#ImageDimensionsAngstrom/pix")
        if vol_a:
            d.add("tomo_dims_px", tuple(round(v / pix) for v in vol_a), "derived", f"{xml_path.name}#VolumeDimensionsAngstrom/pix")

    # --- frames / averages -----------------------------------------------------
    movie_paths = _text("MoviePath")
    while len(movie_paths) > n and movie_paths and movie_paths[-1] == "":
        movie_paths.pop()
    if len(movie_paths) != n:
        movie_paths = [""] * n
    d.facts["movie_paths"] = movie_paths
    tomostar_dir = None
    if s is not None and s.data_folder and settings_path is not None:
        cand = (settings_path.parent / s.data_folder)
        tomostar_dir = cand if cand.is_dir() else None
    if tomostar_dir is None:
        cand = xml_path.parent.parent / "tomostar"
        tomostar_dir = cand if cand.is_dir() else xml_path.parent
    d.facts["tomostar_dir"] = tomostar_dir
    if any(movie_paths):
        d.add("tilt_image_names", [Path(mp.replace("\\", "/")).stem if mp else "" for mp in movie_paths],
              "file", f"{xml_path.name}#MoviePath")
    averages = resolve_average_paths(xml_path, movie_paths, frames_dir=frames_dir, tomostar_dir=tomostar_dir)
    d.facts["average_paths"] = averages
    present = [a for a in averages if a is not None and a.exists()]
    if frames_dir is not None:
        d.add("frames_dir", str(Path(frames_dir)), "cli", "--frames-dir")
    elif present:
        d.add("frames_dir", str(present[0].parent.parent), "derived", "MoviePath relative to the tomostar directory")
    if present:
        h = mrc_header(present[0])
        d.add("image_dims_px", (h["nx"], h["ny"]), "file", f"{present[0].name}#header")
        if h["voxel"][0] > 0:
            d.add("pixel_size_a", round(h["voxel"][0], 6), "file", f"{present[0].name}#header")
    d.facts["n_averages_present"] = len(present)

    # --- stack ---------------------------------------------------------------
    stack = Path(tilt_stack) if tilt_stack else None
    if stack is None and tilt_stack_dir is not None:
        for ext in (".mrc", ".st", ".mrcs"):
            cand = Path(tilt_stack_dir) / f"{stem}{ext}"
            if cand.exists():
                stack = cand
                break
    if stack is not None:
        h = mrc_header(stack)
        d.add("stack_path", str(stack), "file", stack.name)
        d.add("image_dims_px", (h["nx"], h["ny"]), "file", f"{stack.name}#header")
        d.add("n_raw_sections", h["nz"], "file", f"{stack.name}#header")
    dose = _text("Dose")
    if len(dose) == n:
        try:
            d.facts["pre_exposure"] = [float(v) for v in dose]
        except ValueError:
            pass
    return d
