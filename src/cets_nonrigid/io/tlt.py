"""``<stem>_TLT.txt`` for AreTomo3-target conversions.

AreTomo3 ``-Cmd 2`` reads ``<stem>_TLT.txt`` before ``<stem>.rawtlt``
(CTsPackage.cpp mLoadTiltFile) and applies its lines positionally to the
stack slices, so the file must carry one line per RAW section in stack
order: the ``.aln`` global rows in FILE order with ``# DarkFrame`` angles
re-inserted at their raw index. Angles are the refined ``TILT`` values
(AlphaOffset inside) — the CTF correction takes its angles from here.
Acquisition index and per-image dose come from the source's pre-exposure
(``RawDose``); without one a one-column file is written.
"""

from __future__ import annotations

from cryoet_alignment.io.aretomo3 import AreTomo3TLT
from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN
from cryoet_alignment.io.aretomo3.tlt import TltInfo

from cets_nonrigid.io.dose import RawDose


def tlt_angles_from_aln(aln: AreTomo3ALN) -> list[float]:
    """Refined tilt per raw section (darks included), refusing files whose
    SEC order is not the row order (positional application would then
    silently mis-assign rows)."""
    from cets_nonrigid.io.aln import raw_tilts_from_aln

    secs = [int(g.sec) for g in aln.GlobalAlignments]
    if secs != sorted(secs):
        raise ValueError(
            ".aln SEC column is not ascending in row order — AreTomo3 applies rows "
            "positionally; refusing to write a _TLT.txt for it"
        )
    return [float(v) for v in raw_tilts_from_aln(aln).tolist()]


def tlt_from_aln(aln: AreTomo3ALN, raw_dose: RawDose | None = None) -> AreTomo3TLT:
    """Build the ``_TLT.txt`` model: angles from the .aln, acquisition index and
    per-image dose from ``raw_dose`` (per raw row) when given."""
    angles = tlt_angles_from_aln(aln)
    if raw_dose is None:
        return AreTomo3TLT(rows=[TltInfo(tilt=a) for a in angles])
    if raw_dose.n_raw != len(angles):
        raise ValueError(f"raw_dose covers {raw_dose.n_raw} rows but the .aln raw stack has {len(angles)}")
    acq = [int(v) for v in raw_dose.acq_index_1b.tolist()]
    dose = [max(0.0, float(v)) for v in raw_dose.dose_per_image.tolist()]
    return AreTomo3TLT(
        rows=[TltInfo(tilt=a, acq_index=i, dose=d) for a, i, d in zip(angles, acq, dose)]
    )
