"""SeriesMeta: what a conversion knows about one tilt series and where each
value came from. Plain optional fields plus a provenance map; anything a
target needs beyond these is a CLI flag."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProvKind = Literal["cli", "file", "derived", "portal", "default", "unset"]

#: Fields a to-RELION / to-Warp conversion may need; used to report defaults.
EXPERIMENTAL_FIELDS = (
    "pixel_size_a",
    "image_dims_px",
    "tomo_dims_px",
    "voltage_kv",
    "cs_mm",
    "amplitude_contrast",
    "defocus_hand",
    "acq_order_1b",
    "dose_per_tilt",
    "tilt_image_names",
    "ctf_path",
)

FIELD_FLAGS = {
    "pixel_size_a": "--pix",
    "tomo_dims_px": "--tomo-size",
    "voltage_kv": "--voltage",
    "cs_mm": "--cs",
    "amplitude_contrast": "--amp-contrast",
    "defocus_hand": "--defocus-hand",
    "acq_order_1b": "a <stem>_TLT.txt (tilt, acquisition index[, dose]) or <stem>.mdoc next to the .aln",
    "dose_per_tilt": "--dose-per-tilt",
    "tilt_image_names": "--frames-dir",
    "ctf_path": "<stem>_CTF.txt next to the .aln",
    "stack_path": "--tilt-stack-dir",
    "angles_inverted": "--angles-inverted",
    "image_dims_px": "--image-size",
}


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProvKind
    source: str
    note: str = ""

    def __str__(self) -> str:
        s = f"{self.kind}:{self.source}"
        return f"{s} ({self.note})" if self.note else s


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    value: Any
    provenance: Provenance


class MetaConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    values: list[str]  # "value <- provenance" per source

    def __str__(self) -> str:
        return f"{self.field}: " + " | ".join(self.values)


class MetaConflictError(ValueError):
    """Two discovered sources disagree beyond tolerance (and no CLI value
    settles it)."""

    def __init__(self, conflicts: list[MetaConflict]):
        self.conflicts = conflicts
        super().__init__(
            "conflicting metadata sources — pass the explicit flag to settle it:\n  "
            + "\n  ".join(str(c) for c in conflicts)
        )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)) and len(value) > 6:
        return f"[{len(value)} values]"
    return str(value)


class SeriesMeta(BaseModel):
    """Resolved per-series metadata with provenance."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    series_name: str
    source: str
    pixel_size_a: float | None = None
    image_dims_px: tuple[int, int] | None = None
    n_raw_sections: int | None = None
    tomo_dims_px: tuple[int, int, int] | None = None
    voltage_kv: float | None = None
    cs_mm: float | None = None
    amplitude_contrast: float | None = None
    defocus_hand: int | None = None
    angles_inverted: bool | None = None
    tilt_axis_deg: float | None = None
    acq_order_1b: list[int] | None = None
    stage_tilt_deg: list[float] | None = None  # per raw row, acquisition stage angles
    dose_per_tilt: float | None = None
    dose_per_section: list[float] | None = None
    dose_convention: str = "exclusive"
    raw_dose: Any | None = None  # cets_nonrigid.io.dose.RawDose
    tilt_image_names: list[str] | None = None
    frames_dir: str | None = None
    stack_path: str | None = None
    ctf_path: str | None = None
    ctf_source: str = "none"
    aux: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, Provenance] = Field(default_factory=dict)
    conflicts: list[MetaConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def get_provenance(self, field: str) -> Provenance:
        return self.provenance.get(field, Provenance(kind="unset", source=""))

    def defaulted(self) -> list[str]:
        """Experimental fields that no source supplied."""
        out = []
        for f in EXPERIMENTAL_FIELDS:
            if getattr(self, f) is None or self.get_provenance(f).kind in ("default", "unset"):
                out.append(f)
        return out

    def report(self) -> dict:
        """JSON-serialisable: value + provenance per field, plus warnings."""
        values = {}
        for f in type(self).model_fields:
            if f in ("provenance", "conflicts", "warnings", "aux", "raw_dose"):
                continue
            v = getattr(self, f)
            if v is None:
                continue
            values[f] = {"value": v, "provenance": str(self.get_provenance(f))}
        return {
            "series_name": self.series_name,
            "source": self.source,
            "fields": values,
            "aux": dict(self.aux),
            "defaulted": self.defaulted(),
            "warnings": list(self.warnings),
        }

    def summary_line(self) -> str:
        parts = []
        for f in ("pixel_size_a", "image_dims_px", "tomo_dims_px", "voltage_kv", "cs_mm",
                  "amplitude_contrast", "defocus_hand", "dose_per_tilt"):
            v = getattr(self, f)
            if v is not None:
                parts.append(f"{f}={_fmt(v)} <- {self.get_provenance(f)}")
        if self.acq_order_1b is not None:
            parts.append(f"acq_order <- {self.get_provenance('acq_order_1b')}")
        if self.ctf_path is not None:
            parts.append(f"ctf <- {self.get_provenance('ctf_path')}")
        if self.stack_path is not None:
            parts.append(f"stack <- {self.get_provenance('stack_path')}")
        if self.tilt_image_names is not None:
            parts.append(f"tilt_images[{len(self.tilt_image_names)}] <- {self.get_provenance('tilt_image_names')}")
        return f"{self.series_name}: " + "; ".join(parts)
