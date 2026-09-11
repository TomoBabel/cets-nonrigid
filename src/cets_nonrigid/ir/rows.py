"""Row identity between a pipeline-agnostic IR and a fit target (plan rev. 5).

The IR's ``/projection`` table is the authority on its own rows; a target
(template XML / template .aln) brings its own row table. ``match_rows``
resolves the correspondence with an explicit priority:

1. an explicit ``--row-map`` file (zero-based ``target_idx source_idx`` lines),
2. stable labels — participating only when nonempty and unique on BOTH
   selected row sets,
3. angle matching as a CHECKED fallback with an explicit compatibility
   matrix: nominal<->nominal and effective<->effective are eligible;
   ``unknown`` on either side is never eligible; everything else demands
   labels or a row map. Angles must additionally be unique within tolerance
   on both sides; angle values are compared 1:1 (the cets_nonrigid ecosystem
   pins matching angle columns across formats — a differing convention is
   exactly what the label/row-map escape hatches are for).

Every selected (valid) source row must map exactly once; unmatched ACTIVE
target rows fail unless ``deactivate_unmatched``. ``align_ir_rows`` then
permutes/expands every T-indexed IR array into target row order — including
both held-out sets and both global baselines — synthesizing invalid rows for
unmatched targets, and ALWAYS replaces the row metadata with the target row
table (identity maps short-circuit array copying only).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from cets_nonrigid.ir.core import IRMeta, IRTiltSeries, row_labels_from_paths

DEFAULT_ANGLE_TOL_DEG = 0.5
_AMBIGUITY_MARGIN_DEG = 0.05


@dataclass
class TargetRowTable:
    """The target's own rows: everything align_ir_rows needs to write correct
    metadata for matched AND unmatched rows."""

    angle_deg: list  # (T,) target-native angle column
    angle_kind: list  # per row: nominal | effective | unknown
    active: list  # UseTilt / non-dark regular row
    dark: list
    sec: list  # raw section numbers (-1 = unknown)
    dose: list  # cumulative e/A^2 (0.0 = unknown)
    labels: list | None = None  # stable identity, or None
    order: list | None = None  # acquisition order (None -> range(T))

    def __post_init__(self):
        t = len(self.angle_deg)
        for name in ("angle_kind", "active", "dark", "sec", "dose"):
            if len(getattr(self, name)) != t:
                raise ValueError(f"TargetRowTable.{name} has {len(getattr(self, name))} rows for {t}")
        if self.labels is not None and len(self.labels) != t:
            raise ValueError("TargetRowTable.labels length mismatch")
        if self.order is None:
            self.order = list(range(t))

    @property
    def n_rows(self) -> int:
        return len(self.angle_deg)

    @classmethod
    def from_warp_ts(cls, ts) -> TargetRowTable:
        """From a warpylib TiltSeries (template XML). Warp Angles are not
        provenance-proven nominal stage metadata -> kind 'unknown'."""
        use = [bool(u) for u in ts.use_tilt]
        return cls(
            angle_deg=[float(a) for a in ts.angles],
            angle_kind=["unknown"] * ts.n_tilts,
            active=use,
            dark=[not u for u in use],
            sec=[-1] * ts.n_tilts,
            dose=[float(d) for d in ts.dose],
            labels=row_labels_from_paths(getattr(ts, "tilt_movie_paths", None)),
            order=None,
        )

    @classmethod
    def from_template_aln(cls, aln) -> TargetRowTable:
        """From a template .aln: raw-section rows incl. DarkFrame records,
        preserving the template's SEC numbering. TILT is refined -> 'effective'."""
        from cets_nonrigid.io.aln import raw_tilts_from_aln

        tilts = raw_tilts_from_aln(aln)
        r = tilts.shape[0]
        dark = [False] * r
        for d in aln.DarkFrames or []:
            dark[int(d.section_idx)] = True
        return cls(
            angle_deg=[float(a) for a in tilts],
            angle_kind=["effective"] * r,
            active=[not d for d in dark],
            dark=dark,
            sec=list(range(1, r + 1)),  # .aln SEC is the 1-based raw index
            dose=[0.0] * r,
            labels=None,
            order=None,
        )

    @classmethod
    def identity_of_ir(cls, meta: IRMeta) -> TargetRowTable:
        """The IR's own rows as the target (template-free fits)."""
        t = len(meta.projection_index or [])
        valid = list(meta.projection_valid or [True] * t)
        dark = list(meta.projection_dark or [False] * t)
        return cls(
            angle_deg=list(meta.projection_angle_deg or [0.0] * t),
            angle_kind=list(meta.projection_angle_kind or ["unknown"] * t),
            active=valid,
            dark=dark,
            sec=list(meta.projection_sec or [-1] * t),
            dose=list(meta.projection_dose or [0.0] * t),
            labels=meta.projection_label,
            order=list(meta.projection_order or range(t)),
        )


@dataclass
class RowMatch:
    row_map: list  # source (IR) row -> target row, for SELECTED source rows only; -1 elsewhere
    target_active: list  # per target row, after any deactivation
    method: str  # row_map | labels | angles | identity


def _read_row_map_file(path: str | Path, n_source: int, n_target: int) -> list:
    """Zero-based ``target_idx source_idx`` lines -> source->target map."""
    row_map = [-1] * n_source
    seen_targets: set[int] = set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"row-map line {lineno}: expected 'target_idx source_idx'")
        tgt, src = int(parts[0]), int(parts[1])
        if not (0 <= tgt < n_target) or not (0 <= src < n_source):
            raise ValueError(
                f"row-map line {lineno}: indices ({tgt}, {src}) out of range "
                f"(target < {n_target}, source < {n_source}; indices are ZERO-based)"
            )
        if row_map[src] != -1 or tgt in seen_targets:
            raise ValueError(f"row-map line {lineno}: duplicate source or target index")
        row_map[src] = tgt
        seen_targets.add(tgt)
    return row_map


def _usable_labels(labels, selected) -> dict | None:
    if labels is None:
        return None
    chosen = {labels[i]: i for i in selected}
    if any(not labels[i] for i in selected) or len(chosen) != len(selected):
        return None  # must be nonempty and unique on the SELECTED set
    return chosen


_KIND_ELIGIBLE = {("nominal", "nominal"), ("effective", "effective")}


def match_rows(
    meta: IRMeta,
    target: TargetRowTable,
    *,
    row_map_file: str | Path | None = None,
    angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
    deactivate_unmatched: bool = False,
) -> RowMatch:
    t_src = len(meta.projection_index or [])
    src_valid = list(meta.projection_valid or [True] * t_src)
    selected = [i for i in range(t_src) if src_valid[i]]
    tgt_active = list(target.active)

    if row_map_file is not None:
        row_map = _read_row_map_file(row_map_file, t_src, target.n_rows)
        method = "row_map"
    else:
        src_labels = _usable_labels(meta.projection_label, selected)
        tgt_selectable = [j for j in range(target.n_rows)]
        tgt_labels = _usable_labels(target.labels, tgt_selectable)
        if src_labels is not None and tgt_labels is not None:
            row_map = [-1] * t_src
            for label, i in src_labels.items():
                if label not in tgt_labels:
                    raise ValueError(
                        f"source row {i} (label {label!r}) has no target counterpart - "
                        "source data is never silently dropped"
                    )
                row_map[i] = tgt_labels[label]
            method = "labels"
        else:
            row_map = _match_by_angles(meta, target, selected, angle_tol_deg)
            method = "angles"

    # every selected source row maps exactly once
    mapped = [row_map[i] for i in selected]
    if any(m < 0 for m in mapped):
        missing = [i for i in selected if row_map[i] < 0]
        raise ValueError(f"selected source rows {missing} are unmapped")
    if len(set(mapped)) != len(mapped):
        raise ValueError("row map is not injective on the selected source rows")

    unmatched_active = [j for j in range(target.n_rows) if tgt_active[j] and j not in set(mapped)]
    if unmatched_active:
        if not deactivate_unmatched:
            raise ValueError(
                f"target rows {unmatched_active} are active but matched by no source row - "
                "pass deactivate_unmatched to disable them instead"
            )
        for j in unmatched_active:
            tgt_active[j] = False

    if row_map == list(range(t_src)) and target.n_rows == t_src:
        method = "identity"
    return RowMatch(row_map=row_map, target_active=tgt_active, method=method)


def _match_by_angles(meta, target, selected, tol_deg) -> list:
    src_kinds = meta.projection_angle_kind or ["unknown"] * len(meta.projection_index or [])
    src_angles = meta.projection_angle_deg or []

    def check_unique(values, side):
        import itertools

        vals = sorted(values)
        for a, b in itertools.pairwise(vals):
            if b - a < tol_deg:
                raise ValueError(
                    f"{side} angles are not unique within {tol_deg} deg "
                    f"({a:.3f} vs {b:.3f}) - automatic angle matching is unsafe; "
                    "provide row labels or an explicit --row-map"
                )

    for i in selected:
        if src_kinds[i] == "unknown":
            raise ValueError(
                f"source row {i} has angle_kind 'unknown' - automatic angle matching "
                "is not eligible; provide row labels or an explicit --row-map"
            )
    tgt_candidates = [j for j in range(target.n_rows)]
    for j in tgt_candidates:
        if target.angle_kind[j] == "unknown":
            raise ValueError(
                f"target row {j} has angle_kind 'unknown' - automatic angle matching "
                "is not eligible; provide row labels or an explicit --row-map"
            )
    kinds = {(src_kinds[i], target.angle_kind[j]) for i in selected for j in tgt_candidates}
    bad = kinds - _KIND_ELIGIBLE
    if bad:
        raise ValueError(
            f"angle-kind combination(s) {sorted(bad)} are not eligible for automatic "
            "matching (only nominal<->nominal and effective<->effective are); "
            "provide row labels or an explicit --row-map"
        )
    check_unique([src_angles[i] for i in selected], "source")
    check_unique([target.angle_deg[j] for j in tgt_candidates], "target")

    row_map = [-1] * len(src_kinds)
    taken: set[int] = set()
    for i in selected:
        errs = sorted(
            (abs(src_angles[i] - target.angle_deg[j]), j)
            for j in tgt_candidates
            if j not in taken
        )
        if not errs or errs[0][0] > tol_deg:
            closest = errs[0][0] if errs else float("inf")
            raise ValueError(
                f"source row {i} (angle {src_angles[i]:.2f} deg) matches no target row "
                f"within {tol_deg} deg (closest: {closest:.2f}) - source data is never "
                "silently dropped"
            )
        if len(errs) > 1 and errs[1][0] - errs[0][0] < _AMBIGUITY_MARGIN_DEG:
            raise ValueError(f"ambiguous angle match for source row {i}")
        row_map[i] = errs[0][1]
        taken.add(errs[0][1])
    return row_map


def align_ir_rows(ir: IRTiltSeries, match: RowMatch, target: TargetRowTable) -> IRTiltSeries:
    """Permute/expand every T-indexed array into target row order and REPLACE
    the row metadata with the target row table (also on identity maps)."""
    t_tgt = target.n_rows
    row_map = match.row_map

    meta = ir.meta.model_copy(
        update={
            "projection_index": list(range(t_tgt)),
            "projection_valid": [bool(a) for a in match.target_active],
            "projection_order": list(target.order),
            "projection_dose": [float(d) for d in target.dose],
            "projection_angle_deg": [float(a) for a in target.angle_deg],
            "projection_sec": [int(s) for s in target.sec],
            "projection_dark": [bool(d) for d in target.dark],
            "projection_label": list(target.labels) if target.labels is not None else None,
            "projection_angle_kind": [str(k) for k in target.angle_kind],
        }
    )

    identity = row_map == list(range(ir.n_projections)) and t_tgt == ir.n_projections

    def expand(arr: torch.Tensor, fill=0) -> torch.Tensor:
        if identity:
            return arr
        out = torch.full(
            (t_tgt, *arr.shape[1:]), fill, dtype=arr.dtype
        ) if arr.dtype != torch.bool else torch.zeros((t_tgt, *arr.shape[1:]), dtype=torch.bool)
        for src, tgt in enumerate(row_map):
            if tgt >= 0:
                out[tgt] = arr[src]
        return out

    active = torch.tensor(match.target_active, dtype=torch.bool)
    projection_valid = expand(ir.projection_valid) & active[:, None]
    heldout_projection_valid = expand(ir.heldout_projection_valid) & active[:, None]

    return replace(
        ir,
        source_projected=expand(ir.source_projected),
        source_projected_global=expand(ir.source_projected_global),
        projection_valid=projection_valid,
        weights=expand(ir.weights),
        heldout_source_projected=expand(ir.heldout_source_projected),
        heldout_projection_valid=heldout_projection_valid,
        heldout_weights=expand(ir.heldout_weights),
        heldout_source_projected_global=(
            expand(ir.heldout_source_projected_global)
            if ir.heldout_source_projected_global is not None
            else None
        ),
        # schema 0.4 optional arrays: unmatched target rows are inactive; fill 0
        # keeps the all-finite contract (validity lives in the row table)
        source_displacement_3d=(
            expand(ir.source_displacement_3d) if ir.source_displacement_3d is not None else None
        ),
        heldout_source_displacement_3d=(
            expand(ir.heldout_source_displacement_3d)
            if ir.heldout_source_displacement_3d is not None
            else None
        ),
        source_ctf_depth_a=(expand(ir.source_ctf_depth_a) if ir.source_ctf_depth_a is not None else None),
        heldout_source_ctf_depth_a=(
            expand(ir.heldout_source_ctf_depth_a) if ir.heldout_source_ctf_depth_a is not None else None
        ),
        meta=meta,
    )
