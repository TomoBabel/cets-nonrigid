"""Acquisition-order + dose sources for the AreTomo3-source conversions.

AreTomo3 sorts the raw stack ascending by tilt angle and resets section
indices (CProcessThread.cpp:207-213), so per-tilt pre-exposure needs the
ACQUISITION order from one of:

  * an mdoc (per-ZValue TiltAngle/ExposureDose/DateTime; parsed with the
    ``mdocfile`` package, entries sorted by acquisition time);
  * an AreTomo3 three-column ``_TLT.txt`` (tilt, 1-based acquisition index,
    per-image dose — CTsPackage.cpp mSaveTiltFile; parsed with
    ``cryoet_alignment.io.aretomo3.AreTomo3TLT``);
  * an explicit acquisition-order file (one 1-based acquisition index per raw
    ascending-tilt row, darks included) plus --dose-per-tilt;
  * the clearly named unsafe assumption that stack order == acquisition
    order, plus --dose-per-tilt.

Pre-exposure convention (``rlnMicrographPreExposure``): **exclusive** by
default — the dose the specimen received BEFORE this image (0 for the first
acquired tilt; the portal ``accumulated_dose``, ZPT, Warp ``<Dose>`` and
RELION's own ``dose_per_tilt_image`` branch, tomography_python_programs/
_utils/mdoc.py:30). ``convention="inclusive"`` reproduces RELION's mdoc
``ExposureDose`` import branch (np.cumsum, mdoc.py:28), one image dose higher
everywhere. RELION's C++ treats the column as an opaque per-frame scalar
(tomogram.cpp:221,294), so both are valid; a project must use one.

Pre-exposure is always computed over the COMPLETE raw acquisition sequence
(including subsequently dropped dark tilts) and only then subset to emitted
rows. Missing or all-zero dose is an error, never silent zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch

_F64 = torch.float64

DoseConvention = Literal["exclusive", "inclusive"]
DEFAULT_DOSE_CONVENTION: DoseConvention = "exclusive"

_MDOC_DATETIME_FORMATS = ("%d-%b-%Y  %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%b-%y  %H:%M:%S", "%d-%b-%y %H:%M:%S")


def _check_convention(convention: str) -> str:
    if convention not in ("exclusive", "inclusive"):
        raise ValueError(f"dose convention must be 'exclusive' or 'inclusive', got {convention!r}")
    return convention


@dataclass
class RawDose:
    """Per RAW-SORTED-row (ascending AreTomo tilt, darks included) metadata."""

    tilt_deg: torch.Tensor  # (R,)
    acq_index_1b: torch.Tensor  # (R,) 1-based acquisition order, unique
    pre_exposure: torch.Tensor  # (R,) cumulative e/A^2 in the requested convention
    convention: str = DEFAULT_DOSE_CONVENTION

    def __post_init__(self) -> None:
        r = self.tilt_deg.shape[0]
        acq = self.acq_index_1b
        if acq.shape != (r,) or self.pre_exposure.shape != (r,):
            raise ValueError("inconsistent RawDose array lengths")
        vals = sorted(int(a) for a in acq)
        if any(a != round(a) for a in acq.tolist()) or vals != list(range(1, r + 1)):
            raise ValueError(
                f"acquisition indices must be unique integers 1..{r}, got {acq.tolist()}"
            )
        if not torch.isfinite(self.pre_exposure).all() or (self.pre_exposure < 0).any():
            raise ValueError("pre-exposure doses must be finite and non-negative")
        _check_convention(self.convention)

    @property
    def n_raw(self) -> int:
        return self.tilt_deg.shape[0]

    @property
    def dose_per_image(self) -> torch.Tensor:
        """Per raw row: the dose this image itself received (increments of the
        cumulative sequence; the last acquired image repeats the previous
        increment when the sequence is exclusive)."""
        order = torch.argsort(self.acq_index_1b)
        cum = self.pre_exposure[order].to(_F64)
        n = cum.shape[0]
        frac = torch.empty(n, dtype=_F64)
        if self.convention == "inclusive":
            frac[0] = cum[0]
            frac[1:] = cum[1:] - cum[:-1]
        else:
            if n > 1:
                frac[:-1] = cum[1:] - cum[:-1]
                frac[-1] = frac[-2]
            else:
                frac[0] = 0.0
        out = torch.empty_like(frac)
        out[order] = frac
        return out


def _pre_exposure_from_fractional(
    acq_index_1b: torch.Tensor,
    dose_frac: torch.Tensor,
    *,
    convention: str = DEFAULT_DOSE_CONVENTION,
) -> torch.Tensor:
    """Cumulative dose in acquisition order, returned per raw row. Inclusive
    mirrors RELION's np.cumsum(ExposureDose); exclusive subtracts each image's
    own dose (dose BEFORE the image)."""
    _check_convention(convention)
    order = torch.argsort(acq_index_1b)
    frac = dose_frac[order].to(_F64)
    cum = torch.cumsum(frac, dim=0)
    if convention == "exclusive":
        cum = cum - frac
    out = torch.empty_like(cum)
    out[order] = cum
    return out


def _mdoc_time_key(entry: dict):
    """Chronological sort key: parsed DateTime, else the ZValue (SerialEM
    writes sections in acquisition order)."""
    text = entry.get("DateTime")
    if text:
        s = str(text).strip()
        for fmt in _MDOC_DATETIME_FORMATS:
            try:
                # SerialEM stamps carry no timezone; only the ORDER matters here
                return (0, datetime.strptime(s, fmt).timestamp(), int(entry["ZValue"]))  # noqa: DTZ007
            except ValueError:
                continue
        return (1, s, int(entry["ZValue"]))
    return (2, "", int(entry["ZValue"]))


def parse_mdoc(path: str | Path) -> list:
    """mdoc reader (``mdocfile``): one dict per [ZValue] section, in file order,
    with the section's non-empty fields plus ``ZValue``."""
    from mdocfile.data_models import Mdoc

    mdoc = Mdoc.from_file(str(path))
    entries = []
    for section in mdoc.section_data:
        d = {k: v for k, v in section.model_dump().items() if v is not None}
        if section.ZValue is None:
            continue
        d["ZValue"] = int(section.ZValue)
        entries.append(d)
    if not entries:
        raise ValueError(f"{path}: no [ZValue] sections")
    return entries


def raw_dose_from_mdoc(
    path: str | Path,
    raw_tilt_deg: torch.Tensor,  # (R,) raw ascending-tilt rows (darks included)
    *,
    dose_per_tilt: float | None = None,
    tilt_match_tol_deg: float = 0.5,
    convention: str = DEFAULT_DOSE_CONVENTION,
) -> RawDose:
    _check_convention(convention)
    entries = parse_mdoc(path)
    if len(entries) != raw_tilt_deg.shape[0]:
        raise ValueError(
            f"{path}: {len(entries)} mdoc sections for {raw_tilt_deg.shape[0]} raw tilts"
        )
    ordered = sorted(entries, key=_mdoc_time_key)
    if dose_per_tilt is not None:
        # RELION's dose_per_tilt_image branch: dose * arange(len) (exclusive)
        k = torch.arange(len(ordered), dtype=_F64)
        pre = float(dose_per_tilt) * (k if convention == "exclusive" else k + 1)
    else:
        doses = [float(e.get("ExposureDose", "nan")) for e in ordered]
        d = torch.tensor(doses, dtype=_F64)
        if not torch.isfinite(d).all() or float(d.abs().sum()) == 0.0:
            raise ValueError(
                f"{path}: no usable ExposureDose values — pass an explicit "
                "--dose-per-tilt override (silent zero dose is refused)"
            )
        pre = torch.cumsum(d, dim=0)
        if convention == "exclusive":
            pre = pre - d

    # match mdoc tilts to raw-sorted rows by angle (bijection, tolerance).
    # A CONSTANT offset between the two lists is allowed: with -TiltCor the
    # .aln TILT column includes AlphaOffset while the mdoc carries raw stage
    # angles — remove the mean difference before matching.
    mdoc_tilt = torch.tensor([float(e["TiltAngle"]) for e in ordered], dtype=_F64)
    raw = raw_tilt_deg.to(_F64)
    offset = float(raw.mean() - mdoc_tilt.mean())
    mdoc_tilt = mdoc_tilt + offset
    used = torch.zeros(len(ordered), dtype=torch.bool)
    acq = torch.empty(raw_tilt_deg.shape[0], dtype=torch.long)
    pre_row = torch.empty(raw_tilt_deg.shape[0], dtype=_F64)
    for r, tilt in enumerate(raw):
        diff = (mdoc_tilt - tilt).abs() + used.to(_F64) * 1e6
        j = int(torch.argmin(diff))
        if float(diff[j]) > tilt_match_tol_deg:
            raise ValueError(
                f"{path}: no mdoc tilt within {tilt_match_tol_deg} deg of raw tilt "
                f"{float(tilt):.2f} (closest: {float(mdoc_tilt[j]):.2f})"
            )
        used[j] = True
        acq[r] = j + 1
        pre_row[r] = pre[j]
    return RawDose(
        tilt_deg=raw_tilt_deg.to(_F64), acq_index_1b=acq, pre_exposure=pre_row, convention=convention
    )


def raw_dose_from_tlt(
    path: str | Path,
    *,
    dose_per_tilt: float | None = None,
    convention: str = DEFAULT_DOSE_CONVENTION,
) -> RawDose:
    """Three-column AreTomo3 _TLT.txt (raw ascending-tilt row order). A
    one-column IMOD .tlt carries no acquisition order and is rejected."""
    from cryoet_alignment.io.aretomo3 import AreTomo3TLT

    _check_convention(convention)
    tlt = AreTomo3TLT.from_file(str(path))
    if not tlt.has_acq_index:
        raise ValueError(
            f"{path}: expected the three-column AreTomo3 _TLT.txt "
            "(tilt, acquisition index, dose); got a one-column tilt list"
        )
    tilt = torch.tensor(tlt.tilts, dtype=_F64)
    acq = torch.tensor(tlt.acq_indices, dtype=torch.long)
    if dose_per_tilt is not None:
        frac = torch.full((tlt.n_rows,), float(dose_per_tilt), dtype=_F64)
    else:
        frac = torch.tensor(tlt.doses, dtype=_F64) if tlt.has_dose else torch.zeros(tlt.n_rows, dtype=_F64)
        if not torch.isfinite(frac).all() or float(frac.abs().sum()) == 0.0:
            raise ValueError(
                f"{path}: no usable dose column — pass an explicit --dose-per-tilt "
                "override (silent zero dose is refused)"
            )
    return RawDose(
        tilt_deg=tilt, acq_index_1b=acq,
        pre_exposure=_pre_exposure_from_fractional(acq, frac, convention=convention),
        convention=convention,
    )


def raw_dose_from_acq_order(
    path: str | Path,
    raw_tilt_deg: torch.Tensor,
    *,
    dose_per_tilt: float,
    convention: str = DEFAULT_DOSE_CONVENTION,
) -> RawDose:
    """Explicit file: one 1-based acquisition index per raw ascending-tilt row
    (darks included). Requires --dose-per-tilt (the file carries no dose)."""
    vals = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(vals) != raw_tilt_deg.shape[0]:
        raise ValueError(f"{path}: {len(vals)} indices for {raw_tilt_deg.shape[0]} raw tilts")
    acq = torch.tensor([int(v) for v in vals], dtype=torch.long)
    return raw_dose_from_indices(raw_tilt_deg, acq, dose_per_tilt=dose_per_tilt, convention=convention)


def raw_dose_from_indices(
    raw_tilt_deg: torch.Tensor,
    acq_index_1b: torch.Tensor,
    *,
    dose_per_tilt: float | None = None,
    dose_per_image: torch.Tensor | None = None,
    convention: str = DEFAULT_DOSE_CONVENTION,
) -> RawDose:
    """Acquisition indices already known (metadata discovery, portal API):
    per-image dose is either a constant or an explicit (R,) tensor."""
    r = raw_tilt_deg.shape[0]
    if dose_per_image is not None:
        frac = dose_per_image.to(_F64)
        if frac.shape != (r,):
            raise ValueError("dose_per_image must have one value per raw row")
    elif dose_per_tilt is not None:
        frac = torch.full((r,), float(dose_per_tilt), dtype=_F64)
    else:
        raise ValueError("need dose_per_tilt or dose_per_image")
    return RawDose(
        tilt_deg=raw_tilt_deg.to(_F64), acq_index_1b=acq_index_1b.to(torch.long),
        pre_exposure=_pre_exposure_from_fractional(acq_index_1b, frac, convention=convention),
        convention=convention,
    )


def raw_dose_assume_stack_order(
    raw_tilt_deg: torch.Tensor,
    *,
    dose_per_tilt: float,
    convention: str = DEFAULT_DOSE_CONVENTION,
) -> RawDose:
    """UNSAFE: assumes the raw-sorted stack order IS the acquisition order."""
    r = raw_tilt_deg.shape[0]
    acq = torch.arange(1, r + 1, dtype=torch.long)
    return raw_dose_from_indices(raw_tilt_deg, acq, dose_per_tilt=dose_per_tilt, convention=convention)


def raw_dose_from_pre_exposure(
    tilt_deg: torch.Tensor,
    pre_exposure: torch.Tensor,
    *,
    convention: str | None = None,
) -> RawDose:
    """Per-row pre-exposure already known (Warp ``<Dose>``, RELION
    ``rlnMicrographPreExposure``): acquisition index = rank of the pre-exposure
    (stable for ties). The convention is detected when not given: a sequence
    whose minimum is 0 is exclusive, otherwise inclusive."""
    pre = pre_exposure.to(_F64)
    if convention is None:
        convention = "exclusive" if float(pre.min()) == 0.0 else "inclusive"
    order = sorted(range(pre.shape[0]), key=lambda i: (float(pre[i]), i))
    acq = torch.empty(pre.shape[0], dtype=torch.long)
    for rank, i in enumerate(order):
        acq[i] = rank + 1
    return RawDose(tilt_deg=tilt_deg.to(_F64), acq_index_1b=acq, pre_exposure=pre, convention=convention)
