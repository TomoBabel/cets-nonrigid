"""Precedence and conflict rules shared by every discoverer.

Precedence: cli > explicit file > stem-discovered file > project file >
portal > default. Every source that produced a value is kept; a disagreement
beyond tolerance is an error listing all values and provenances, unless the
winner is a CLI value (then a warning). Dose is guarded: a per-image dose
read from a local file is used only when it is constant across the series
(coefficient of variation < 5 %), because a tilt-dependent ExposureDose is
the signature of a post-specimen dose reading; portal doses are trusted;
``--dose-per-tilt`` always wins.
"""

from __future__ import annotations

from typing import Any

import torch

from cets_nonrigid.meta.model import (
    FIELD_FLAGS,
    Candidate,
    MetaConflict,
    MetaConflictError,
    Provenance,
    SeriesMeta,
)

_F64 = torch.float64

DOSE_CONSTANT_CV = 0.05

#: (kind, tolerance) per field; "exact" compares equality after normalisation.
TOLERANCES: dict[str, tuple] = {
    "pixel_size_a": ("rel", 1e-4),
    "image_dims_px": ("exact",),
    "n_raw_sections": ("exact",),
    "tomo_dims_px": ("exact",),
    "voltage_kv": ("rel", 1e-3),
    "cs_mm": ("rel", 1e-3),
    "amplitude_contrast": ("abs", 1e-3),
    "defocus_hand": ("exact",),
    "angles_inverted": ("exact",),
    "tilt_axis_deg": ("abs", 0.5),
    "acq_order_1b": ("exact",),
    "stage_tilt_deg": ("abs", 0.5),
    "dose_per_tilt": ("rel", 1e-3),
    "dose_per_section": ("rel", 1e-3),
    "tilt_image_names": ("exact",),
    "ctf_path": ("exact",),
    "stack_path": ("exact",),
    "frames_dir": ("exact",),
}


def _norm(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    return value


def values_agree(field: str, a: Any, b: Any) -> bool:
    spec = TOLERANCES.get(field, ("exact",))
    if spec[0] == "exact":
        return _norm(a) == _norm(b)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        a_t, b_t = torch.tensor(a, dtype=_F64), torch.tensor(b, dtype=_F64)
        if a_t.shape != b_t.shape:
            return False
        diff = (a_t - b_t).abs()
        if spec[0] == "rel":
            return bool((diff <= spec[1] * a_t.abs().clamp(min=1e-12)).all())
        return bool((diff <= spec[1]).all())
    a_f, b_f = float(a), float(b)
    if spec[0] == "rel":
        return abs(a_f - b_f) <= spec[1] * max(abs(a_f), 1e-12)
    return abs(a_f - b_f) <= spec[1]


class Discovered:
    """Candidates collected by a discoverer, in precedence order per field."""

    def __init__(self, series_name: str, source: str):
        self.series_name = series_name
        self.source = source
        self.candidates: dict[str, list[Candidate]] = {}
        self.facts: dict[str, Any] = {}  # non-negotiable data (tensors etc.)
        self.aux: dict[str, str] = {}
        self.warnings: list[str] = []

    def add(self, field: str, value: Any, kind: str, source: str, note: str = "") -> None:
        if value is None:
            return
        self.candidates.setdefault(field, []).append(
            Candidate(value=value, provenance=Provenance(kind=kind, source=source, note=note))
        )

    def first(self, field: str) -> Any:
        c = self.candidates.get(field)
        return c[0].value if c else None


def _cv(values: list[float]) -> float:
    t = torch.tensor(values, dtype=_F64)
    if t.numel() < 2 or float(t.mean()) == 0.0:
        return 0.0
    return float(t.std(unbiased=False) / t.mean().abs())


def resolve_series(
    disc: Discovered,
    cli: dict[str, Any] | None = None,
    *,
    dose_from: str | None = None,
    dose_convention: str = "exclusive",
) -> SeriesMeta:
    """Apply precedence + conflict rules; compose ``raw_dose`` when possible."""
    cli = {k: v for k, v in (cli or {}).items() if v is not None}
    meta = SeriesMeta(series_name=disc.series_name, source=disc.source, dose_convention=dose_convention)
    meta.aux.update(disc.aux)
    meta.warnings.extend(disc.warnings)
    conflicts: list[MetaConflict] = []

    fields = list(disc.candidates.keys()) + [k for k in cli if k not in disc.candidates]
    for field in fields:
        cands = disc.candidates.get(field, [])
        if field in cli:
            value = cli[field]
            prov = Provenance(kind="cli", source=FIELD_FLAGS.get(field, field))
            losers = [c for c in cands if not values_agree(field, value, c.value)]
            if losers:
                meta.warnings.append(
                    f"{field}: {prov.source} {value!r} overrides "
                    + ", ".join(f"{c.value!r} <- {c.provenance}" for c in losers)
                )
        else:
            chosen = cands[0]
            losers = [c for c in cands[1:] if not values_agree(field, chosen.value, c.value)]
            if losers:
                conflicts.append(
                    MetaConflict(
                        field=field,
                        values=[f"{chosen.value!r} <- {chosen.provenance}"]
                        + [f"{c.value!r} <- {c.provenance}" for c in losers],
                    )
                )
                continue
            value, prov = chosen.value, chosen.provenance
        if field in SeriesMeta.model_fields:
            setattr(meta, field, value)
            meta.provenance[field] = prov
    if conflicts:
        meta.conflicts = conflicts
        raise MetaConflictError(conflicts)

    _resolve_dose(meta, disc, cli, dose_from=dose_from, dose_convention=dose_convention)
    return meta


def _resolve_dose(meta: SeriesMeta, disc: Discovered, cli: dict, *, dose_from, dose_convention) -> None:
    """Dose guard + RawDose composition (needs the raw tilt list fact)."""
    from cets_nonrigid.io.dose import raw_dose_from_indices

    per_section = None
    if "dose_per_tilt" in cli:
        pass  # explicit constant, already set with cli provenance
    else:
        # candidates for per-section doses, in precedence order; portal trusted,
        # local files only when constant unless forced by dose_from
        for cand in disc.candidates.get("dose_per_section", []):
            src_kind = cand.provenance.kind
            src_tag = cand.provenance.note or cand.provenance.source
            forced = dose_from is not None and (dose_from == "file" or dose_from in src_tag)
            if src_kind == "portal" or forced or _cv(cand.value) < DOSE_CONSTANT_CV:
                per_section = cand.value
                meta.dose_per_section = per_section
                meta.provenance["dose_per_section"] = cand.provenance
                if _cv(per_section) < DOSE_CONSTANT_CV:
                    meta.dose_per_tilt = float(torch.tensor(per_section, dtype=_F64).mean())
                    meta.provenance["dose_per_tilt"] = Provenance(
                        kind="derived", source=cand.provenance.source, note="constant per-image dose"
                    )
                break
            meta.warnings.append(
                f"dose: per-image doses in {cand.provenance.source} vary by "
                f"{100 * _cv(cand.value):.0f}% (looks post-specimen) — not used; pass "
                "--dose-per-tilt D, or --dose-per-tilt file to use them as they are"
            )
        if per_section is None and meta.dose_per_tilt is None:
            meta.provenance.setdefault("dose_per_tilt", Provenance(kind="unset", source=""))

    raw_tilts = disc.facts.get("raw_tilts")
    if meta.stage_tilt_deg is not None:
        raw_tilts = torch.tensor(meta.stage_tilt_deg, dtype=_F64)
    elif disc.facts.get("stage_tilts_fallback") is not None:
        raw_tilts = disc.facts["stage_tilts_fallback"]
        meta.stage_tilt_deg = [float(v) for v in raw_tilts.tolist()]
        meta.provenance["stage_tilt_deg"] = Provenance(kind="derived", source="aln#TILT-AlphaOffset")
    if raw_tilts is not None and meta.acq_order_1b is not None:
        acq = torch.tensor(meta.acq_order_1b, dtype=torch.long)
        if "dose_per_tilt" in cli or (per_section is None and meta.dose_per_tilt is not None):
            meta.raw_dose = raw_dose_from_indices(
                raw_tilts, acq, dose_per_tilt=meta.dose_per_tilt, convention=dose_convention
            )
        elif per_section is not None:
            meta.raw_dose = raw_dose_from_indices(
                raw_tilts, acq, dose_per_image=torch.tensor(per_section, dtype=_F64),
                convention=dose_convention,
            )
