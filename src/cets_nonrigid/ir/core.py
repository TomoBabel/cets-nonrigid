"""Transient containers shared by the transferred numerical fitters.

This module is internal, not an exchange schema. At the CETS boundary,
``source_projected_global`` and its held-out twin are evaluated from the document
in float64; ``source_projected`` is reconstructed as G + r. Runtime views use the
validated native corner frame. Neither global array nor IRMeta is persisted.

Validity remains explicit; fit weights never encode availability. Optional 3D
displacement and signed CTF depth are independent channels. Target-local fits
subtract their own global/premovement baseline. The projected observations already
include displacement, so its projected contribution must never be added twice.

The native builders retain optional legacy-state labels internally for compatibility
with transferred kernel tests. No legacy Zarr reader or writer is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict


def row_labels_from_paths(paths) -> list | None:
    """Stable per-row labels from per-tilt image/movie paths: the basenames,
    but only when every row has one and they are unique (the matcher's
    participation requirement); None otherwise."""
    if not paths:
        return None
    names = [str(p).replace("\\", "/").rsplit("/", 1)[-1] for p in paths]
    if any(not n for n in names) or len(set(names)) != len(names):
        return None
    return names


class IRMeta(BaseModel):
    """Transient metadata derived from CETS for the native corner-coordinate fitters."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tilt_series", "frame_series"]
    series_name: str = ""
    frame: str = "warp-canonical"
    pixel_size_image_a: float
    image_dims_px: tuple[int, int]
    # Tilt series only:
    volume_dims_px: tuple[int, int, int] | None = None
    pixel_size_volume_a: float | None = None
    # Per-projection table (Warp file order incl. darks for tilt series):
    projection_index: list[int] = []
    projection_valid: list[bool] = []
    projection_order: list[int] = []
    projection_dose: list[float] = []
    # Tilt series only (frame-series stores omit these):
    projection_angle_deg: list[float] | None = None
    projection_sec: list[int] | None = None  # -1 sentinel where unknown
    projection_dark: list[bool] | None = None
    source_tool: str = ""
    #: "grid": regular training grid + Sobol held-out (grid_shape set);
    #: "particles": scattered particle positions, held-out = particle subset
    #: (grid_shape None; point names/indices stored as arrays, not here).
    sampling: Literal["grid", "particles"] = "grid"
    #: Optional stable per-row identity (image basename / movie path) where the
    #: source provides one — enables label-based row matching.
    projection_label: list[str] | None = None
    #: What projection_angle_deg MEANS, per row: "nominal" (stage metadata),
    #: "effective" (refined), "unknown". Only provenance-proven kinds are set;
    #: automatic angle matching is restricted to compatible non-unknown kinds.
    projection_angle_kind: list[str] | None = None
    #: Why ``source_displacement_3d`` is present or absent (module docstring).
    displacement_3d: Literal["present", "zero_at_samples", "none", "not_recorded"] = "none"
    #: Why ``source_ctf_depth_a`` is present or absent.
    source_ctf_depth: Literal["present", "none", "not_recorded"] = "none"


@dataclass
class IRTiltSeries:
    grid_shape: tuple[int, int, int] | None  # None for particle sampling
    points: torch.Tensor  # (N, 3) f8, A, canonical volume frame
    source_projected: torch.Tensor  # (T, N, 2) f4, A, canonical image frame
    source_projected_global: torch.Tensor  # (T, N, 2) f4
    sample_valid: torch.Tensor  # (N,) bool
    projection_valid: torch.Tensor  # (T, N) bool
    weights: torch.Tensor  # (T, N) f4
    heldout_points: torch.Tensor  # (M, 3) f8; M == 0 -> heldout not evaluated
    heldout_source_projected: torch.Tensor  # (T, M, 2) f4
    heldout_projection_valid: torch.Tensor  # (T, M) bool
    heldout_weights: torch.Tensor  # (T, M) f4
    #: None = 0.2 store; (T,0,2)/(T,M,2) = 0.3 (see module docstring)
    native_model: object | None = field(default=None, kw_only=True, repr=False)
    native_data: object | None = field(default=None, kw_only=True, repr=False)
    heldout_source_projected_global: torch.Tensor | None = field(default=None, kw_only=True)
    meta: IRMeta = field(kw_only=True)
    # Particle sampling only (stored as arrays in the store, never in attrs):
    point_names: list[str] | None = field(default=None, kw_only=True)
    heldout_point_names: list[str] | None = field(default=None, kw_only=True)
    point_index: torch.Tensor | None = field(default=None, kw_only=True)  # (N,) i8 original indices
    heldout_point_index: torch.Tensor | None = field(default=None, kw_only=True)  # (M,) i8
    # Schema 0.4 optional arrays (both of a pair are set or both None; see module docstring):
    source_displacement_3d: torch.Tensor | None = field(default=None, kw_only=True)  # (T, N, 3) f4, A
    heldout_source_displacement_3d: torch.Tensor | None = field(default=None, kw_only=True)  # (T, M, 3) f4
    source_ctf_depth_a: torch.Tensor | None = field(default=None, kw_only=True)  # (T, N) f4, A
    heldout_source_ctf_depth_a: torch.Tensor | None = field(default=None, kw_only=True)  # (T, M) f4

    @property
    def n_projections(self) -> int:
        return self.source_projected.shape[0]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def heldout_status(self) -> str:
        """'evaluated' | 'not_evaluated' — consumers must branch on this
        before touching any held-out metric or array."""
        return "evaluated" if self.heldout_points.shape[0] > 0 else "not_evaluated"

    def warped_points(self) -> torch.Tensor | None:
        """(T, N, 3) f8 ``points + source_displacement_3d``; None when absent."""
        if self.source_displacement_3d is None:
            return None
        return self.points.to(torch.float64)[None] + self.source_displacement_3d.to(torch.float64)

    def heldout_warped_points(self) -> torch.Tensor | None:
        if self.heldout_source_displacement_3d is None:
            return None
        return self.heldout_points.to(torch.float64)[None] + self.heldout_source_displacement_3d.to(torch.float64)

    def validate_optional_arrays(self) -> None:
        """Shape/finiteness/flag consistency of the 0.4 optional arrays."""
        t, n, m = self.n_projections, self.n_points, self.heldout_points.shape[0]
        d, hd = self.source_displacement_3d, self.heldout_source_displacement_3d
        if (d is None) != (hd is None):
            raise ValueError("source_displacement_3d and its held-out twin must both be set or both None")
        if d is not None:
            if d.shape != (t, n, 3) or hd.shape != (t, m, 3):
                raise ValueError(
                    f"displacement shapes {tuple(d.shape)}/{tuple(hd.shape)} != ({t}, {n}, 3)/({t}, {m}, 3)"
                )
            if not (torch.isfinite(d).all() and torch.isfinite(hd).all()):
                raise ValueError("source_displacement_3d contains non-finite values")
            if self.meta.displacement_3d != "present":
                raise ValueError(f"displacement arrays present but meta.displacement_3d={self.meta.displacement_3d!r}")
        elif self.meta.displacement_3d == "present":
            raise ValueError("meta.displacement_3d='present' without arrays")
        c, hc = self.source_ctf_depth_a, self.heldout_source_ctf_depth_a
        if (c is None) != (hc is None):
            raise ValueError("source_ctf_depth_a and its held-out twin must both be set or both None")
        if c is not None:
            if c.shape != (t, n) or hc.shape != (t, m):
                raise ValueError(f"ctf-depth shapes {tuple(c.shape)}/{tuple(hc.shape)} != ({t}, {n})/({t}, {m})")
            if not (torch.isfinite(c).all() and torch.isfinite(hc).all()):
                raise ValueError("source_ctf_depth_a contains non-finite values")
            if self.meta.source_ctf_depth != "present":
                raise ValueError(f"ctf-depth arrays present but meta.source_ctf_depth={self.meta.source_ctf_depth!r}")
        elif self.meta.source_ctf_depth == "present":
            raise ValueError("meta.source_ctf_depth='present' without arrays")


@dataclass
class IRFrameSeries:
    grid_shape: tuple[int, int]
    points: torch.Tensor  # (N, 2) f8
    source_projected: torch.Tensor  # (F, N, 2) f4
    source_projected_global: torch.Tensor  # (F, N, 2) f4
    sample_valid: torch.Tensor  # (N,) bool
    projection_valid: torch.Tensor  # (F, N) bool
    weights: torch.Tensor  # (F, N) f4
    heldout_points: torch.Tensor  # (M, 2) f8
    heldout_source_projected: torch.Tensor  # (F, M, 2) f4
    heldout_projection_valid: torch.Tensor  # (F, M) bool
    heldout_weights: torch.Tensor  # (F, M) f4
    #: None = 0.2 store; (T,0,2)/(T,M,2) = 0.3 (see module docstring)
    native_model: object | None = field(default=None, kw_only=True, repr=False)
    native_data: object | None = field(default=None, kw_only=True, repr=False)
    heldout_source_projected_global: torch.Tensor | None = field(default=None, kw_only=True)
    meta: IRMeta = field(kw_only=True)

    @property
    def n_projections(self) -> int:
        return self.source_projected.shape[0]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def heldout_status(self) -> str:
        return "evaluated" if self.heldout_points.shape[0] > 0 else "not_evaluated"
