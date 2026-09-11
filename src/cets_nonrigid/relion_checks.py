"""Machine-readable gates for RELION bundles / projects (pass, fail,
not_evaluated) — the checks the validation campaigns re-implemented."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cets_nonrigid.project.common import Gate, gate

_F64 = torch.float64


def _image_ref_exists(name: str, root: Path) -> bool | None:
    """``N@path`` or ``path``: True/False when checkable, None for a placeholder
    reference that is not meant to exist as real pixels."""
    part = name.split("@", 1)[1] if "@" in name else name
    if "placeholder" in Path(part).name:
        return None
    p = Path(part)
    return (p if p.is_absolute() else root / p).exists()


def check_series(
    *,
    name: str,
    tilt_star: Path,
    tomo,  # RelionTomogramData as emitted
    root: Path,
    n_particles: int = 0,
    motion: dict | None = None,
    require_images: bool = True,
) -> list[Gate]:
    from cets_nonrigid.io.relion_star import read_star_tables

    gates: list[Gate] = []
    blocks = read_star_tables(tilt_star)
    ok_block = name in blocks
    gates.append(gate("tilt_star_block", ok_block, value=list(blocks)[:2], expected=f"data_{name}"))
    if not ok_block:
        return gates
    df = blocks[name]
    gates.append(gate("rows_emitted", len(df) == tomo.n_tilts, value=len(df), expected=tomo.n_tilts))
    mandatory = ["rlnTomoYTilt", "rlnTomoZRot", "rlnTomoXShiftAngst", "rlnTomoYShiftAngst",
                 "rlnMicrographPreExposure", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle"]
    missing = [c for c in mandatory if c not in df.columns]
    gates.append(gate("mandatory_columns", not missing, value=missing, expected="all present"))
    if not missing:
        got = np.stack([np.asarray(df[c], dtype=np.float64) for c in
                        ("rlnTomoYTilt", "rlnTomoZRot", "rlnTomoXShiftAngst", "rlnTomoYShiftAngst")], axis=1)
        want = torch.stack([tomo.ytilt_deg, tomo.zrot_deg, tomo.xshift_a, tomo.yshift_a], dim=1).to(_F64).numpy()
        if got.shape == want.shape:
            dev = float(np.abs(got - want).max())
            gates.append(gate("write_read_fidelity", dev <= 1e-5, value=dev, expected="<= 1e-5 (starfile %.6f)"))
        nominal = np.asarray(df["rlnTomoNominalStageTiltAngle"], dtype=np.float64) if "rlnTomoNominalStageTiltAngle" in df.columns else None
        if nominal is not None:
            distinct = len(np.unique(np.round(nominal, 3))) == len(nominal)
            gates.append(gate("nominal_angles_distinct", distinct, value=len(np.unique(np.round(nominal, 3))),
                              expected=len(nominal), note="template-free r2w matches rows by nominal angle"))
        pre = np.asarray(df["rlnMicrographPreExposure"], dtype=np.float64)
        gates.append(gate("pre_exposure_finite", bool(np.isfinite(pre).all()), value=bool(np.isfinite(pre).all()),
                          expected=True))
    if "rlnMicrographName" in df.columns:
        names = [str(x) for x in df["rlnMicrographName"]]
        checks = [_image_ref_exists(n, root) for n in names]
        if all(c is None for c in checks):
            gates.append(gate("images_exist", None, value="placeholder", expected="skipped for placeholder stacks"))
        elif require_images:
            missing_imgs = [n for n, c in zip(names, checks) if c is False]
            gates.append(gate("images_exist", not missing_imgs, value=len(missing_imgs), expected=0,
                              note=missing_imgs[0] if missing_imgs else ""))
    if n_particles and motion is not None:
        bad_rows = [k for k, v in motion.items() if len(v) != tomo.n_tilts]
        gates.append(gate("motion_blocks", len(motion) == n_particles and not bad_rows,
                          value=(len(motion), len(bad_rows)), expected=(n_particles, 0)))
    return gates


def check_project_dose_convention(pre_exposure_min_by_series: dict) -> Gate:
    """Every series exclusive (min pre-exposure == 0 over the complete
    sequence) or every series inclusive — never mixed."""
    if not pre_exposure_min_by_series:
        return gate("dose_convention_uniform", None, expected="no series")
    kinds = {name: ("exclusive" if abs(v) < 1e-9 else "inclusive") for name, v in pre_exposure_min_by_series.items()}
    uniform = len(set(kinds.values())) == 1
    return gate("dose_convention_uniform", uniform, value=sorted(set(kinds.values())),
                expected="one convention across the project",
                note="" if uniform else ", ".join(f"{k}:{v}" for k, v in kinds.items()))
