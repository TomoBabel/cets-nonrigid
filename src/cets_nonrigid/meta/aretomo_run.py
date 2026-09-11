"""Discover a series' metadata from the files AreTomo3 leaves next to its
``.aln`` (stem-adjacent, header-only reads), plus the mdoc when present.

Never consulted: ``AreTomo3_Session.json`` (user decision). Defocus hand is
never derived — ``_CTF.txt`` dfHand semantics are unpinned.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from cets_nonrigid.meta.resolve import Discovered

_F64 = torch.float64
_TILT_AXIS_RE = re.compile(r"TiltAxisAngle\s*=\s*([-+]?\d+(?:\.\d+)?)")


def mrc_header(path: Path) -> dict:
    """nx, ny, nz, voxel (A) and mode from an MRC header only."""
    import mrcfile

    with mrcfile.open(str(path), header_only=True, permissive=True) as m:
        h = m.header
        vs = m.voxel_size
        return {
            "nx": int(h.nx), "ny": int(h.ny), "nz": int(h.nz), "mode": int(h.mode),
            "voxel": (float(vs.x), float(vs.y), float(vs.z)),
        }


def _find_stack(stem: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        for ext in (".mrc", ".st", ".mrcs"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def mdoc_acquisition(mdoc_path: Path, raw_tilts: torch.Tensor, *, tol_deg: float = 0.5):
    """Per raw row: 1-based acquisition index and the mdoc section, matched by
    stage angle after removing the mean offset (AlphaOffset)."""
    from cets_nonrigid.io.dose import _mdoc_time_key, parse_mdoc

    entries = parse_mdoc(mdoc_path)
    if len(entries) != raw_tilts.shape[0]:
        raise ValueError(f"{mdoc_path}: {len(entries)} sections for {raw_tilts.shape[0]} raw tilts")
    ordered = sorted(entries, key=_mdoc_time_key)
    mtilt = torch.tensor([float(e["TiltAngle"]) for e in ordered], dtype=_F64)
    raw = raw_tilts.to(_F64)
    mtilt = mtilt + float(raw.mean() - mtilt.mean())
    used = torch.zeros(len(ordered), dtype=torch.bool)
    acq, per_row = [], []
    for tilt in raw:
        diff = (mtilt - tilt).abs() + used.to(_F64) * 1e6
        j = int(torch.argmin(diff))
        if float(diff[j]) > tol_deg:
            raise ValueError(
                f"{mdoc_path}: no section within {tol_deg} deg of raw tilt {float(tilt):.2f}"
            )
        used[j] = True
        acq.append(j + 1)
        per_row.append(ordered[j])
    return acq, per_row


def _sub_frame_stem(value) -> str:
    text = str(value).replace("\\", "/")
    return Path(text).stem


def discover_aretomo_series(
    aln_path: str | Path,
    *,
    mdoc: str | Path | None = None,
    mdoc_dir: str | Path | None = None,
    tilt_stack: str | Path | None = None,
    tilt_stack_dir: str | Path | None = None,
    tlt: str | Path | None = None,
    ctf: str | Path | None = None,
    no_ctf: bool = False,
    discover_adjacent: bool = True,
) -> Discovered:
    """``discover_adjacent=False`` consults only the explicitly given files
    (--no-discover); the .aln header is always read."""
    from cryoet_alignment.io.aretomo3 import AreTomo3ALN, AreTomo3TLT

    from cets_nonrigid.io.aln import raw_tilts_from_aln

    aln_path = Path(aln_path)
    stem = aln_path.stem
    aln = AreTomo3ALN.from_file(str(aln_path))
    raw_tilts = raw_tilts_from_aln(aln)
    n_raw = int(aln.RawSize[2])
    alpha = float(aln.AlphaOffset or 0.0)

    d = Discovered(series_name=stem, source=str(aln_path))
    # stage angles per raw row: .aln TILT carries AlphaOffset on global rows only
    # (DarkFrame records store the raw stage angle) — the derived fallback when
    # no _TLT.txt / mdoc supplies acquisition stage angles
    stage_tilts = raw_tilts.clone().to(_F64)
    for g in aln.GlobalAlignments:
        stage_tilts[int(g.sec) - 1] = float(g.tilt) - alpha
    d.facts.update({
        "aln": aln, "raw_tilts": raw_tilts, "alpha_offset_deg": alpha, "stem": stem,
        "stage_tilts_fallback": stage_tilts,  # used only when no file supplies stage angles
    })
    d.add("image_dims_px", (int(aln.RawSize[0]), int(aln.RawSize[1])), "file", f"{aln_path.name}#RawSize")
    d.add("n_raw_sections", n_raw, "file", f"{aln_path.name}#RawSize")
    if aln.GlobalAlignments:
        d.add("tilt_axis_deg", float(aln.GlobalAlignments[0].rot), "derived", f"{aln_path.name}#ROT")

    # --- tilt stack (pixel size + dims) --------------------------------------
    stack = Path(tilt_stack) if tilt_stack else None
    stack_kind = "file"
    if stack is None:
        dirs = [Path(tilt_stack_dir)] if tilt_stack_dir else []
        if discover_adjacent:
            dirs.append(aln_path.parent)
        stack = _find_stack(stem, dirs) if dirs else None
        stack_kind = "file"
    if stack is not None:
        h = mrc_header(stack)
        d.add("stack_path", str(stack), stack_kind, stack.name)
        d.add("image_dims_px", (h["nx"], h["ny"]), "file", f"{stack.name}#header")
        d.add("n_raw_sections", h["nz"], "file", f"{stack.name}#header")
        if h["voxel"][0] > 0:
            d.add("pixel_size_a", round(h["voxel"][0], 6), "file", f"{stack.name}#header")
        d.facts["stack_header"] = h

    # --- _TLT.txt: acquisition order + per-image dose --------------------------
    tlt_path = Path(tlt) if tlt else (aln_path.with_name(f"{stem}_TLT.txt") if discover_adjacent else None)
    if tlt_path is not None and tlt_path.exists():
        t = AreTomo3TLT.from_file(str(tlt_path))
        if t.n_rows != n_raw:
            d.warnings.append(f"{tlt_path.name}: {t.n_rows} rows for {n_raw} raw sections — ignored")
        else:
            d.aux["tlt"] = str(tlt_path)
            d.add("stage_tilt_deg", [float(v) for v in t.tilts], "file", tlt_path.name, note="tlt")
            if t.has_acq_index:
                d.add("acq_order_1b", list(t.acq_indices), "file", tlt_path.name, note="tlt")
            if t.has_dose and sum(t.doses) > 0:
                d.add("dose_per_section", list(t.doses), "file", tlt_path.name, note="tlt")
            stage = raw_tilts.to(_F64) - alpha
            dev = (torch.tensor(t.tilts, dtype=_F64) - stage).abs().max()
            if float(dev) > 0.5:
                d.warnings.append(
                    f"{tlt_path.name}: tilt column deviates from .aln stage angles by up to {float(dev):.2f} deg"
                )

    # --- mdoc: pixel size, voltage, order, dose, movie names --------------------
    mdoc_path = Path(mdoc) if mdoc else None
    if mdoc_path is None:
        adjacent = [aln_path.with_name(f"{stem}.mdoc")] if discover_adjacent else []
        for cand in ([Path(mdoc_dir) / f"{stem}.mdoc"] if mdoc_dir else []) + adjacent:
            if cand.exists():
                mdoc_path = cand
                break
    if mdoc_path is not None and mdoc_path.exists():
        from mdocfile.data_models import Mdoc

        m = Mdoc.from_file(str(mdoc_path))
        d.aux["mdoc"] = str(mdoc_path)
        g = m.global_data
        pix = g.PixelSpacing if g.PixelSpacing else (m.section_data[0].PixelSpacing if m.section_data else None)
        if pix:
            d.add("pixel_size_a", float(pix), "file", f"{mdoc_path.name}#PixelSpacing")
        volt = g.Voltage if g.Voltage else (m.section_data[0].Voltage if m.section_data else None)
        if volt:
            d.add("voltage_kv", float(volt), "file", f"{mdoc_path.name}#Voltage")
        if g.ImageSize:
            d.add("image_dims_px", (int(g.ImageSize[0]), int(g.ImageSize[1])), "file", f"{mdoc_path.name}#ImageSize")
        for title in m.titles or []:
            mt = _TILT_AXIS_RE.search(title)
            if mt:
                d.aux["mdoc_tilt_axis_angle"] = mt.group(1)
        try:
            acq, rows = mdoc_acquisition(mdoc_path, raw_tilts)
        except ValueError as e:
            d.warnings.append(str(e))
        else:
            d.add("acq_order_1b", acq, "file", mdoc_path.name, note="mdoc")
            d.add("stage_tilt_deg", [float(r["TiltAngle"]) for r in rows], "file", mdoc_path.name, note="mdoc")
            doses = [float(r.get("ExposureDose", 0.0) or 0.0) for r in rows]
            if sum(doses) > 0:
                d.add("dose_per_section", doses, "file", f"{mdoc_path.name}#ExposureDose", note="mdoc")
            if all(r.get("SubFramePath") for r in rows):
                d.add(
                    "tilt_image_names", [_sub_frame_stem(r["SubFramePath"]) for r in rows],
                    "file", f"{mdoc_path.name}#SubFramePath",
                )

    # --- CTF ---------------------------------------------------------------
    if not no_ctf:
        ctf_path = Path(ctf) if ctf else (aln_path.with_name(f"{stem}_CTF.txt") if discover_adjacent else None)
        if ctf_path is not None and ctf_path.exists():
            d.add("ctf_path", str(ctf_path), "file", ctf_path.name)

    # --- tomogram Z from <stem>_Vol.mrc (header only) --------------------------
    vol = aln_path.with_name(f"{stem}_Vol.mrc")
    if discover_adjacent and vol.exists():
        try:
            h = mrc_header(vol)
            rx, ry = int(aln.RawSize[0]), int(aln.RawSize[1])
            b = rx / h["nx"] if h["nx"] else 0.0
            if b > 0:
                if abs(h["ny"] * b - ry) <= b:  # thickness along z (FlipVol 1)
                    z = h["nz"] * b
                elif abs(h["nz"] * b - ry) <= b:  # thickness along y (FlipVol 0)
                    z = h["ny"] * b
                else:
                    z = None
                if z is not None:
                    d.add("tomo_dims_px", (rx, ry, round(z)), "derived",
                          f"{vol.name}#header x bin {b:g}")
        except Exception as e:  # noqa: BLE001 - header trouble is only a lost candidate
            d.warnings.append(f"{vol.name}: unreadable header ({e})")
    return d
