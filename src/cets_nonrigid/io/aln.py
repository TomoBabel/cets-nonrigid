"""AreTomo3 ``.aln`` adapter and tilt matching.

Parsing itself is delegated to ``cryoet_alignment.io.aretomo3.aln`` (the
pydantic model); this module enforces the real-output invariants, builds the
torch :class:`AretomoTsModel`, and matches ``.aln`` rows (dark-removed,
tilt-ascending) to Warp XML tilts (file order) by stage angle.

AreTomo3 only — AreTomo2 files are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

from cets_nonrigid.models.aretomo_ts import AreTomoLocalField, AretomoTsModel

ANGLE_MATCH_TOL_DEG = 0.5


def raw_tilts_from_aln(aln: AreTomo3ALN) -> torch.Tensor:
    """(R,) raw-sorted tilt list including darks, reconstructed from the .aln
    global rows (SEC is the 1-based raw-sorted index) and # DarkFrame lines."""
    r = int(aln.RawSize[2])
    tilts = torch.full((r,), float("nan"), dtype=torch.float64)
    for g in aln.GlobalAlignments:
        tilts[int(g.sec) - 1] = float(g.tilt)
    for d in aln.DarkFrames or []:
        tilts[int(d.section_idx)] = float(d.angle)
    if not torch.isfinite(tilts).all():
        missing = torch.nonzero(~torch.isfinite(tilts)).flatten().tolist()
        raise ValueError(f".aln does not cover raw sections {missing} (SEC/DarkFrame mismatch)")
    return tilts


@dataclass
class AlnSeries:
    aln: AreTomo3ALN
    model: AretomoTsModel
    aln_bytes: bytes
    stage_angles_deg: torch.Tensor  # (T,) TILT - AlphaOffset


@dataclass
class TiltMatch:
    """Bijection between .aln rows and Warp tilt indices."""

    aln_to_warp: list[int]  # index into Warp file order, per .aln row
    unmatched_warp: list[int]  # Warp tilts with no .aln counterpart (darks etc.)
    max_angle_error_deg: float


def load_aln(
    path: str | Path,
    *,
    pixel_size_a: float,
    volume_dims_a: tuple[float, float, float],
) -> AlnSeries:
    """Load an AreTomo3 .aln and build the projection model.

    pixel_size_a: pixel size of the motion-corrected tilt series (the scale of
        all .aln pixel quantities).
    volume_dims_a: canonical tomogram box (Angstrom) the model speaks.
    """
    path = Path(path)
    aln_bytes = path.read_bytes()
    aln = AreTomo3ALN.from_file(str(path))

    t = len(aln.GlobalAlignments)
    if t == 0:
        raise ValueError(f"{path}: no global alignment rows")
    p = aln.NumPatches

    if p > 0:
        n_local = len(aln.LocalAlignments or [])
        if n_local != t * p:
            raise ValueError(
                f"{path}: local section has {n_local} rows, expected "
                f"{t} tilts x {p} patches = {t * p} (truncated or non-AreTomo3 file?)"
            )

    g = np.array([[e.rot, e.tilt, e.tx, e.ty] for e in aln.GlobalAlignments], dtype=np.float64)
    nx, ny = int(aln.RawSize[0]), int(aln.RawSize[1])

    local = None
    if p > 0:
        coord = np.zeros((t, p, 2))
        shift = np.zeros((t, p, 2))
        good = np.zeros((t, p))
        seen = np.zeros((t, p), dtype=bool)
        for row in aln.LocalAlignments:
            ti, pi = int(row.sec_idx), int(row.patch_idx)
            if not (0 <= ti < t and 0 <= pi < p):
                raise ValueError(f"{path}: local row indices ({ti}, {pi}) out of range")
            if seen[ti, pi]:
                raise ValueError(f"{path}: duplicate local row ({ti}, {pi})")
            seen[ti, pi] = True
            coord[ti, pi] = (row.center_x, row.center_y)
            shift[ti, pi] = (row.shift_x, row.shift_y)
            good[ti, pi] = row.is_reliable
        local = AreTomoLocalField(
            coord_xy=torch.tensor(coord),
            shift_xy=torch.tensor(shift),
            good=torch.tensor(good),
            raw_size_px=torch.tensor([nx, ny]),
        )

    model = AretomoTsModel(
        rot_deg=torch.tensor(g[:, 0]),
        tilt_deg=torch.tensor(g[:, 1]),
        shifts_px=torch.tensor(g[:, 2:4]),
        raw_size_px=(nx, ny),
        pixel_size_a=pixel_size_a,
        volume_dims_a=volume_dims_a,
        local=local,
    )
    stage = torch.tensor(g[:, 1]) - float(aln.AlphaOffset)
    return AlnSeries(aln=aln, model=model, aln_bytes=aln_bytes, stage_angles_deg=stage)


def match_tilts(
    warp_angles_deg: torch.Tensor,
    warp_use_tilt: torch.Tensor,
    aln_series: AlnSeries,
    tol_deg: float = ANGLE_MATCH_TOL_DEG,
) -> TiltMatch:
    """Match .aln rows to Warp tilts by stage angle (bijection required).

    Warp stage angle = -Angle (WARP_TILT_ANGLE_SIGN); .aln stage angle =
    TILT - AlphaOffset. Only Warp tilts with UseTilt=True participate.
    """
    warp_stage = (-warp_angles_deg).to(torch.float64)
    usable = [i for i in range(len(warp_stage)) if bool(warp_use_tilt[i])]
    aln_stage = aln_series.stage_angles_deg

    taken: set[int] = set()
    mapping: list[int] = []
    max_err = 0.0
    for t in range(len(aln_stage)):
        errs = [(abs(float(aln_stage[t] - warp_stage[i])), i) for i in usable if i not in taken]
        if not errs:
            raise ValueError(f"no Warp tilt left to match .aln row {t}")
        err, best = min(errs)
        runners = sorted(e for e, _ in errs)
        if err > tol_deg:
            raise ValueError(
                f".aln row {t} (stage {float(aln_stage[t]):.2f} deg) has no Warp tilt "
                f"within {tol_deg} deg (closest: {err:.2f} deg)"
            )
        if len(runners) > 1 and runners[1] - err < 0.05:
            raise ValueError(
                f".aln row {t}: ambiguous angle match ({err:.3f} vs {runners[1]:.3f} deg) - "
                "duplicate stage angles; SEC-based matching not implemented for this case yet"
            )
        taken.add(best)
        mapping.append(best)
        max_err = max(max_err, err)

    unmatched = [i for i in range(len(warp_stage)) if i not in taken]
    return TiltMatch(aln_to_warp=mapping, unmatched_warp=unmatched, max_angle_error_deg=max_err)


def aln_row_slots(aln: AreTomo3ALN) -> list[int]:
    """Raw-section position -> source image index for a .aln whose global rows
    were written in source order: darks sit at their DarkFrame ``section_idx``
    (raw position) and carry the source index in ``val2``; the remaining raw
    positions take the global rows in file order (``sec - 1``). Identity when
    SEC is dense and there are no darks."""
    r = int(aln.RawSize[2])
    slots: list[int | None] = [None] * r
    for d in aln.DarkFrames or []:
        slots[int(d.section_idx)] = int(d.val2)
    free = [i for i in range(r) if slots[i] is None]
    rows = [int(g.sec) - 1 for g in aln.GlobalAlignments]
    if len(free) != len(rows):
        raise ValueError(f".aln has {len(rows)} global rows for {len(free)} non-dark raw sections")
    for pos, src in zip(free, rows):
        slots[pos] = src
    if sorted(slots) != list(range(r)):
        raise ValueError(".aln SEC/DarkFrame indices are not a permutation of the raw sections")
    return [int(v) for v in slots]


def write_aln(path: str | Path, aln: AreTomo3ALN, **check_kwargs):
    """Serialize, re-read the text and run the text-level post-checks
    (``io.aln_check.check_aln``); returns the AlnCheck (never raises on a gate)."""
    from cets_nonrigid.io.aln_check import check_aln

    path = Path(path)
    path.write_text(str(aln))
    return check_aln(path.read_text(), **check_kwargs)


def assemble_aln(
    *,
    model_global,  # AretomoTsModel in .aln row order (globals only)
    local,  # AreTomoLocalField (coord_xy/shift_xy/good) or None
    sec_1b,  # per .aln row: the SEC value to write
    raw_size: tuple,  # (nx, ny, n_raw_sections)
    dark_frames: list,
    alpha_offset: float,
    beta_offset: float,
    thickness: int | None,
) -> AreTomo3ALN:
    """Shared .aln assembly (w2a, r2a, fit --to aretomo): identical value
    formatting everywhere — globals as raw floats, local patch values rounded
    to 2 decimals, is_reliable = good >= 0.9."""
    from cryoet_alignment.io.aretomo3.aln import GlobalAlignmentInfo, LocalAlignmentInfo

    t_aln = int(model_global.rot_deg.shape[0])
    globals_out = [
        GlobalAlignmentInfo(
            sec=int(sec_1b[t]),
            rot=float(model_global.rot_deg[t]),
            tx=float(model_global.shifts_px[t, 0]),
            ty=float(model_global.shifts_px[t, 1]),
            tilt=float(model_global.tilt_deg[t]),
        )
        for t in range(t_aln)
    ]
    locals_out = None
    p_count = 0
    if local is not None:
        p_count = int(local.coord_xy.shape[1])
        locals_out = [
            LocalAlignmentInfo(
                sec_idx=t,
                patch_idx=p,
                center_x=round(float(local.coord_xy[t, p, 0]), 2),
                center_y=round(float(local.coord_xy[t, p, 1]), 2),
                shift_x=round(float(local.shift_xy[t, p, 0]), 2),
                shift_y=round(float(local.shift_xy[t, p, 1]), 2),
                is_reliable=float(local.good[t, p] >= 0.9),
            )
            for t in range(t_aln)
            for p in range(p_count)
        ]
    return AreTomo3ALN(
        RawSize=(int(raw_size[0]), int(raw_size[1]), int(raw_size[2])),
        NumPatches=p_count,
        DarkFrames=dark_frames,
        AlphaOffset=float(alpha_offset),
        BetaOffset=float(beta_offset),
        Thickness=thickness,
        GlobalAlignments=globals_out,
        LocalAlignments=locals_out,
    )
