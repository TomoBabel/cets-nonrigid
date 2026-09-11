"""Canonical per-tilt CTF representation and exact cross-tool value maps.

All three tools evaluate the same functional form with the polar angle
measured from +x toward +y of the FFT array and the LARGER (more underfocused)
defocus in the direction of the astigmatism angle:

  defocus(theta) = (U + V)/2 + (U - V)/2 * cos(2 * (theta - angle))

so ``DfMax/DfMin/azimuth`` (AreTomo3, Angstrom/degrees, _CTF.txt),
``Defocus +- DefocusDelta/2 / DefocusAngle`` (Warp, micrometers/degrees,
CTF.cs:461-512) and ``rlnDefocusU/V/rlnDefocusAngle`` (RELION, Angstrom/
degrees, ctf.h:184-256 + :297-316) interchange with no angle offset:

  U = (Defocus + DefocusDelta/2) * 1e4     (matches Warp's own export,
  V = (Defocus - DefocusDelta/2) * 1e4      ExportParticlesTiltseries.cs:1049-1054;
                                            NOT Star.GetRelionCTF, whose inverse
                                            halves the delta)

Phase-shift units: AreTomo3 _CTF.txt col 5 = RADIANS (CSaveCtfResults.cpp:93);
Warp stores multiples of pi (CTF.cs:194-197); RELION rlnPhaseShift is DEGREES
(metadata_label.h:890, ctf.cpp:239).

Canonical representation: underfocus-positive Angstrom U >= V, angle in
[0, 180) degrees, phase in degrees. ``score``/``res_a`` are optional
diagnostics — never invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_F64 = torch.float64


def canonicalize_astigmatism(
    defocus_u_a: torch.Tensor, defocus_v_a: torch.Tensor, angle_deg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enforce U >= V (swap + rotate angle by 90 deg where violated) and wrap
    the angle into [0, 180). The cos(2*(theta-angle)) form makes both
    operations exact identities of the defocus field."""
    u = defocus_u_a.to(_F64).clone()
    v = defocus_v_a.to(_F64).clone()
    a = angle_deg.to(_F64).clone()
    swap = v > u
    u2 = torch.where(swap, v, u)
    v2 = torch.where(swap, u, v)
    a = torch.where(swap, a + 90.0, a)
    a = torch.remainder(a, 180.0)
    return u2, v2, a


@dataclass
class TiltCtf:
    """Canonical per-tilt CTF for one tilt series (order = caller-defined)."""

    defocus_u_a: torch.Tensor  # (T,) underfocus-positive Angstrom, U >= V
    defocus_v_a: torch.Tensor  # (T,)
    angle_deg: torch.Tensor  # (T,) in [0, 180)
    phase_deg: torch.Tensor  # (T,)
    score: torch.Tensor | None = None  # (T,) optional diagnostic
    res_a: torch.Tensor | None = None  # (T,) optional fit resolution limit
    voltage_kv: float | None = None
    cs_mm: float | None = None
    amplitude_contrast: float | None = None

    def __post_init__(self) -> None:
        t = self.defocus_u_a.shape[0]
        for name in ("defocus_u_a", "defocus_v_a", "angle_deg", "phase_deg"):
            v = getattr(self, name)
            if v.shape != (t,):
                raise ValueError(f"{name} has shape {tuple(v.shape)}, expected ({t},)")
            if not torch.isfinite(v).all():
                raise ValueError(f"{name} contains non-finite values")
        u, v, a = canonicalize_astigmatism(self.defocus_u_a, self.defocus_v_a, self.angle_deg)
        self.defocus_u_a, self.defocus_v_a, self.angle_deg = u, v, a
        self.phase_deg = self.phase_deg.to(_F64)

    @property
    def n_tilts(self) -> int:
        return self.defocus_u_a.shape[0]

    def defocus_at(self, theta_deg: torch.Tensor) -> torch.Tensor:
        """defocus(theta) = mean + dev*cos(2*(theta - angle)) — for tests."""
        mean = (self.defocus_u_a + self.defocus_v_a) / 2
        dev = (self.defocus_u_a - self.defocus_v_a) / 2
        return mean + dev * torch.cos(2 * torch.deg2rad(theta_deg - self.angle_deg))


def aretomo_raw_order_permutation(warp_angles_deg: torch.Tensor) -> torch.Tensor:
    """Warp-XML-order stage angles -> permutation ``perm`` with ``perm[i]`` =
    XML index of the i-th row of AreTomo3's sorted raw stack (darks included).

    AreTomo3 sorts the raw stack ascending by ITS tilt angle
    (CProcessThread.cpp:207-213), and the two tools record opposite signs
    (TILT = -Angle_warp, conventions.WARP_TILT_ANGLE_SIGN) — so AreTomo's
    ascending order is DESCENDING in Warp angles. Verified on real data:
    24jul16a_Position_16_3.aln rows run TILT -45..+45 (SEC 1..31) while the
    same series' Warp XML Angles run +45..-45. Stable for ties."""
    return torch.argsort(-warp_angles_deg.to(_F64), stable=True)


# --- Warp XML grids -----------------------------------------------------------


def warp_has_tilt_ctf(ts) -> bool:
    """True when the tilt-series carries per-tilt CTF grids (fs/ts_ctf output)."""
    g = ts.grid_ctf_defocus
    if g is None:
        return False
    if tuple(g.dimensions) == (1, 1, 1):
        return ts.n_tilts == 1 and float(g.flat_values[0]) != 0.0
    return True


def _warp_grid_values(ts, name: str, t: int) -> torch.Tensor:
    grid = getattr(ts, name)
    dims = tuple(grid.dimensions)
    if dims != (1, 1, t):
        raise ValueError(
            f"Warp {name} has dims {dims}, expected (1, 1, {t}) — "
            "tilt-series CTF grids are per-tilt temporal grids"
        )
    vals = torch.as_tensor(grid.flat_values, dtype=_F64)
    if vals.shape[0] != t:
        raise ValueError(f"Warp {name} has {vals.shape[0]} values, expected {t}")
    return vals


def tiltctf_from_warp(ts) -> TiltCtf | None:
    """Per-tilt CTF from a warpylib TiltSeries (values in XML tilt order).

    Returns None when the series carries no CTF (untouched template grids).
    Micrometers -> Angstrom; Warp phase (multiples of pi) -> degrees.
    """
    if not warp_has_tilt_ctf(ts):
        return None
    t = ts.n_tilts
    defocus_um = _warp_grid_values(ts, "grid_ctf_defocus", t)
    delta_um = _warp_grid_values(ts, "grid_ctf_defocus_delta", t)
    angle_deg = _warp_grid_values(ts, "grid_ctf_defocus_angle", t)
    phase_pi = _warp_grid_values(ts, "grid_ctf_phase", t)
    return TiltCtf(
        defocus_u_a=(defocus_um + delta_um / 2) * 1e4,
        defocus_v_a=(defocus_um - delta_um / 2) * 1e4,
        angle_deg=angle_deg,
        phase_deg=phase_pi * 180.0,
        voltage_kv=float(ts.ctf.voltage),
        cs_mm=float(ts.ctf.cs),
        amplitude_contrast=float(ts.ctf.amplitude),
    )


def tiltctf_to_warp(ts, ctf: TiltCtf) -> None:
    """Write per-tilt CTF grids (1, 1, T) into a warpylib TiltSeries in place
    (XML tilt order expected) and update the representative <CTF> values."""
    from warpylib.cubic_grid import CubicGrid

    t = ts.n_tilts
    if ctf.n_tilts != t:
        raise ValueError(f"CTF has {ctf.n_tilts} tilts, series has {t}")
    defocus_um = (ctf.defocus_u_a + ctf.defocus_v_a) / 2 * 1e-4
    delta_um = (ctf.defocus_u_a - ctf.defocus_v_a) * 1e-4
    ts.grid_ctf_defocus = CubicGrid((1, 1, t), values=defocus_um.to(torch.float32))
    ts.grid_ctf_defocus_delta = CubicGrid((1, 1, t), values=delta_um.to(torch.float32))
    ts.grid_ctf_defocus_angle = CubicGrid((1, 1, t), values=ctf.angle_deg.to(torch.float32))
    ts.grid_ctf_phase = CubicGrid((1, 1, t), values=(ctf.phase_deg / 180.0).to(torch.float32))
    # representative values on the <CTF> element (per-tilt truth is the grids)
    ts.ctf.defocus = float(defocus_um.mean())
    ts.ctf.defocus_delta = float(delta_um.mean())
    ts.ctf.defocus_angle = float(ctf.angle_deg[0]) if t else 0.0
    ts.ctf.phase_shift = float((ctf.phase_deg / 180.0).mean())
    if ctf.voltage_kv is not None:
        ts.ctf.voltage = float(ctf.voltage_kv)
    if ctf.cs_mm is not None:
        ts.ctf.cs = float(ctf.cs_mm)
    if ctf.amplitude_contrast is not None:
        ts.ctf.amplitude = float(ctf.amplitude_contrast)


# --- RELION per-tilt star columns ---------------------------------------------


def tiltctf_to_relion_columns(ctf: TiltCtf) -> dict:
    """Canonical -> RELION per-tilt star columns (label strings verified:
    rlnPhaseShift is the phase label, metadata_label.h:890)."""
    cols = {
        "rlnDefocusU": ctf.defocus_u_a.numpy(),
        "rlnDefocusV": ctf.defocus_v_a.numpy(),
        "rlnDefocusAngle": ctf.angle_deg.numpy(),
        "rlnPhaseShift": ctf.phase_deg.numpy(),
    }
    if ctf.score is not None:
        cols["rlnCtfFigureOfMerit"] = ctf.score.numpy()
    if ctf.res_a is not None:
        cols["rlnCtfMaxResolution"] = ctf.res_a.numpy()
    return cols


def tiltctf_from_relion_columns(cols: dict, *, voltage_kv=None, cs_mm=None, amplitude_contrast=None) -> TiltCtf:
    """RELION per-tilt star columns -> canonical."""

    def col(name):
        import numpy as np

        return torch.tensor(np.array(cols[name], dtype=np.float64))

    u = col("rlnDefocusU")
    phase = col("rlnPhaseShift") if "rlnPhaseShift" in cols else torch.zeros_like(u)
    return TiltCtf(
        defocus_u_a=u,
        defocus_v_a=col("rlnDefocusV"),
        angle_deg=col("rlnDefocusAngle"),
        phase_deg=phase,
        score=col("rlnCtfFigureOfMerit") if "rlnCtfFigureOfMerit" in cols else None,
        res_a=col("rlnCtfMaxResolution") if "rlnCtfMaxResolution" in cols else None,
        voltage_kv=voltage_kv,
        cs_mm=cs_mm,
        amplitude_contrast=amplitude_contrast,
    )


# --- AreTomo3 _CTF.txt --------------------------------------------------------


def tiltctf_from_aretomo(ctf_file, xml_angles_deg: torch.Tensor) -> TiltCtf:
    """AreTomo3 _CTF.txt -> canonical, reordered into XML tilt order.

    The file's rows are ordinals into the raw ascending-tilt-sorted stack
    INCLUDING darks; ``xml_angles_deg`` are the per-tilt stage angles in XML
    order for the same raw stack (count must match exactly, mirroring
    CLoadCtfResults' all-or-nothing check)."""
    t = xml_angles_deg.shape[0]
    if ctf_file.n_rows != t:
        raise ValueError(
            f"_CTF.txt has {ctf_file.n_rows} rows but the series has {t} tilts "
            "(the file must cover the full raw stack including darks)"
        )
    perm = aretomo_raw_order_permutation(xml_angles_deg)  # raw ordinal -> XML index
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(t)
    # value_in_xml_order[j] = row[inv[j]]
    u_raw = torch.tensor([r.df_max_a for r in ctf_file.rows], dtype=_F64)
    v_raw = torch.tensor([r.df_min_a for r in ctf_file.rows], dtype=_F64)
    a_raw = torch.tensor([r.azimuth_deg for r in ctf_file.rows], dtype=_F64)
    p_raw = torch.tensor([r.phase_rad for r in ctf_file.rows], dtype=_F64)
    s_raw = torch.tensor([r.score for r in ctf_file.rows], dtype=_F64)
    r_raw = torch.tensor([r.res_a for r in ctf_file.rows], dtype=_F64)
    return TiltCtf(
        defocus_u_a=u_raw[inv],
        defocus_v_a=v_raw[inv],
        angle_deg=a_raw[inv],
        phase_deg=torch.rad2deg(p_raw[inv]),  # radians in the file (CSaveCtfResults.cpp:93)
        score=s_raw[inv],
        res_a=r_raw[inv],
    )


def tiltctf_to_aretomo(ctf: TiltCtf, xml_angles_deg: torch.Tensor, df_hand: int = 1):
    """Canonical (XML tilt order) -> AreTomo3 _CTF.txt rows (raw ascending
    order). Score/resolution placeholders (0.0 / 999.99) are emitted when the
    canonical values are absent — they are diagnostics, not fit inputs."""
    from .io.ctf_aretomo import AreTomoCtfFile, AreTomoCtfRow

    t = xml_angles_deg.shape[0]
    if ctf.n_tilts != t:
        raise ValueError(f"CTF has {ctf.n_tilts} tilts, series has {t}")
    perm = aretomo_raw_order_permutation(xml_angles_deg)
    rows = []
    for i in range(t):
        j = int(perm[i])
        rows.append(
            AreTomoCtfRow(
                micrograph=i + 1,
                df_max_a=float(ctf.defocus_u_a[j]),
                df_min_a=float(ctf.defocus_v_a[j]),
                azimuth_deg=float(ctf.angle_deg[j]),
                phase_rad=float(torch.deg2rad(ctf.phase_deg[j])),
                score=float(ctf.score[j]) if ctf.score is not None else 0.0,
                res_a=float(ctf.res_a[j]) if ctf.res_a is not None else 999.99,
                df_hand=df_hand,
            )
        )
    return AreTomoCtfFile(rows=rows)
