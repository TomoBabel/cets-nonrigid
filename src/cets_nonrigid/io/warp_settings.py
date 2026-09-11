"""Warp ``.settings`` for a converted project (adapter over
``cryoet_alignment.io.warp.settings.WarpSettings``)."""

from __future__ import annotations

from cryoet_alignment.io.warp.settings import WarpSettings

#: create_settings slots that must agree for series sharing one project
_SLOTS = (
    ("Import", "PixelSize"),
    ("Import", "DataFolder"),
    ("Import", "ProcessingFolder"),
    ("Import", "Extension"),
    ("Import", "DosePerAngstromFrame"),
    ("Tomo", "DimensionsX"),
    ("Tomo", "DimensionsY"),
    ("Tomo", "DimensionsZ"),
)


def settings_for_project(
    *,
    pixel_size_a: float,
    exposure_per_tilt: float | None,
    tomo_dims_px,
    voltage_kv: float | None = None,
    cs_mm: float | None = None,
    amplitude_contrast: float | None = None,
) -> WarpSettings:
    return WarpSettings.create(
        pixel_size_a=float(pixel_size_a),
        exposure_per_tilt=float(exposure_per_tilt) if exposure_per_tilt is not None else 0.0,
        tomo_dims_px=tuple(int(v) for v in tomo_dims_px),
        voltage_kv=voltage_kv, cs_mm=cs_mm, amplitude_contrast=amplitude_contrast,
    )


def settings_conflicts(existing: WarpSettings, wanted: WarpSettings) -> list[str]:
    """Human-readable differences in the slots ts_reconstruct depends on."""
    out = []
    for section, name in _SLOTS:
        a, b = existing.get(section, name), wanted.get(section, name)
        try:
            same = abs(float(a) - float(b)) <= 1e-6 * max(abs(float(a)), 1.0)
        except (TypeError, ValueError):
            same = a == b
        if not same:
            out.append(f"{section}/{name}: project has {a!r}, this series needs {b!r}")
    return out
