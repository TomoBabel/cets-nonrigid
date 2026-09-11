"""Build the IR by forward-projecting sample grids through a source model."""

from __future__ import annotations

import warnings

import torch

from cets_nonrigid.ir.core import IRFrameSeries, IRMeta, IRTiltSeries
from cets_nonrigid.ir.sampling import (
    boundary_ramp_weights,
    heldout_points,
    image_grid,
    volume_grid,
)
from cets_nonrigid.models.base import FrameMotionModel, TiltProjectionModel

DEFAULT_HELDOUT_FACTOR = 0.6
DEFAULT_HELDOUT_SEED = 20260828


def _source_intermediates(model, points: torch.Tensor):
    """(displacement (T,N,3) f4 | None, flag, ctf_depth (T,N) f4 | None, flag).

    Uses the optional protocols ``VolumeDeformingModel`` / ``CtfDepthModel``
    (``models/base.py``). A ``displace_volume`` that returns ``None`` (RELION
    particle set without trajectories) counts as "no 3D model". Exactly-zero
    sampled displacement is suppressed (``zero_at_samples``): nothing to fit,
    and storing zeros would let a fitter report a "fitted" all-zero warp. CTF
    depth availability is independent of the displacement's (zero trajectories
    do not remove RELION's depth convention).
    """
    disp, disp_flag = None, "none"
    fn = getattr(model, "displace_volume", None)
    if callable(fn):
        d = fn(points)
        if d is not None:
            d = d.to(torch.float32)
            if d.shape != (model.n_projections, points.shape[0], 3):
                raise ValueError(f"displace_volume returned {tuple(d.shape)}, expected ({model.n_projections}, {points.shape[0]}, 3)")
            if not torch.isfinite(d).all():
                raise ValueError("source model produced non-finite 3D displacements")
            if bool((d == 0).all()):
                disp_flag = "zero_at_samples"
            else:
                disp, disp_flag = d, "present"
    ctf, ctf_flag = None, "none"
    fn = getattr(model, "ctf_depth", None)
    if callable(fn):
        c = fn(points)
        if c is not None:
            c = c.to(torch.float32)
            if c.shape != (model.n_projections, points.shape[0]):
                raise ValueError(f"ctf_depth returned {tuple(c.shape)}, expected ({model.n_projections}, {points.shape[0]})")
            if not torch.isfinite(c).all():
                raise ValueError("source model produced non-finite CTF depths")
            ctf, ctf_flag = c, "present"
    return disp, disp_flag, ctf, ctf_flag


def _apply_intermediates(ir: IRTiltSeries, train, held) -> IRTiltSeries:
    """Attach the optional arrays + flags; the pair flags must agree."""
    d, dflag, c, cflag = train
    hd, hdflag, hc, hcflag = held
    if dflag != hdflag:
        # zero_at_samples on one set and present on the other: keep the data
        if "present" in (dflag, hdflag):
            d = d if d is not None else torch.zeros(ir.n_projections, ir.n_points, 3)
            hd = hd if hd is not None else torch.zeros(ir.n_projections, ir.heldout_points.shape[0], 3)
            dflag = "present"
        else:
            raise ValueError(f"inconsistent displacement availability: {dflag} vs {hdflag}")
    if cflag != hcflag:
        raise ValueError(f"inconsistent ctf-depth availability: {cflag} vs {hcflag}")
    ir.meta = ir.meta.model_copy(update={"displacement_3d": dflag, "source_ctf_depth": cflag})
    ir.source_displacement_3d, ir.heldout_source_displacement_3d = d, hd
    ir.source_ctf_depth_a, ir.heldout_source_ctf_depth_a = c, hc
    ir.validate_optional_arrays()
    return ir


def build_ir_tilt_series(
    model: TiltProjectionModel,
    volume_dims_a: torch.Tensor,
    image_dims_a: torch.Tensor,
    *,
    meta: IRMeta,
    grid_shape: tuple[int, int, int] = (15, 15, 5),
    heldout_seed: int = DEFAULT_HELDOUT_SEED,
) -> IRTiltSeries:
    """Project a regular training grid and a Sobol held-out set through the
    source model's FULL warping chain (plus its global-only baseline)."""
    points = volume_grid(volume_dims_a, grid_shape)
    n_heldout = max(64, int(points.shape[0] * DEFAULT_HELDOUT_FACTOR))
    held = heldout_points(volume_dims_a, n_heldout, seed=heldout_seed)

    projected, valid = model.project_volume(points)
    projected_global, _ = model.project_volume_global(points)
    held_projected, held_valid = model.project_volume(held)
    held_projected_global, _ = model.project_volume_global(held)

    weights = boundary_ramp_weights(projected, image_dims_a)
    held_weights = boundary_ramp_weights(held_projected, image_dims_a)

    ir = IRTiltSeries(
        native_model=model,
        grid_shape=grid_shape,
        points=points,
        source_projected=projected.to(torch.float32),
        source_projected_global=projected_global.to(torch.float32),
        sample_valid=torch.ones(points.shape[0], dtype=torch.bool),
        projection_valid=valid,
        weights=weights,
        heldout_points=held,
        heldout_source_projected=held_projected.to(torch.float32),
        heldout_projection_valid=held_valid,
        heldout_weights=held_weights,
        heldout_source_projected_global=held_projected_global.to(torch.float32),
        meta=meta,
    )
    return _apply_intermediates(ir, _source_intermediates(model, points), _source_intermediates(model, held))


def _validate_point_names(names, n: int) -> None:
    if len(names) != n:
        raise ValueError(f"{len(names)} point names for {n} points")
    seen = set()
    for name in names:
        if not name or not isinstance(name, str):
            raise ValueError(f"empty/non-string particle name: {name!r}")
        if any(c.isspace() for c in name) or not name.isprintable():
            raise ValueError(
                f"particle name {name!r} is not STAR-safe (whitespace/control characters)"
            )
        if name in seen:
            raise ValueError(
                f"duplicate particle name {name!r} — duplicate trajectory blocks "
                "silently resolve to the last block in RELION"
            )
        seen.add(name)


def build_ir_tilt_series_from_points(
    model: TiltProjectionModel,
    points_a: torch.Tensor,  # (P, 3) canonical corner-origin Angstrom
    image_dims_a: torch.Tensor,
    *,
    meta: IRMeta,
    point_names=None,
    heldout_fraction: float = 0.2,
    heldout_seed: int = DEFAULT_HELDOUT_SEED,
    min_heldout: int = 16,
) -> IRTiltSeries:
    """Build the IR from SCATTERED particle positions (the from-RELION source
    model is only defined at its particles).

    The FULL set is projected once, then split into train/held-out by a seeded
    permutation — held-out is a particle subset, never a Sobol set. When a
    meaningful split is impossible (too few particles) ALL points become
    training points and ``heldout_status`` reports ``not_evaluated`` — never a
    pass. Original particle indices are persisted so outputs can restore the
    input order. All imported points are validated finite and inside the
    volume (the builder GUARANTEES sample validity; no separate
    heldout_sample_valid array exists).
    """
    if meta.sampling != "particles":
        raise ValueError("meta.sampling must be 'particles' for the scattered-point builder")
    pts = points_a.to(torch.float64)
    p_count = pts.shape[0]
    if p_count == 0:
        raise ValueError("no particles")
    if not torch.isfinite(pts).all():
        raise ValueError("particle positions contain non-finite values")
    if meta.volume_dims_px is None or meta.pixel_size_volume_a is None:
        raise ValueError("meta.volume_dims_px and pixel_size_volume_a are required")
    vol_a = torch.tensor(meta.volume_dims_px, dtype=torch.float64) * meta.pixel_size_volume_a
    if ((pts < 0) | (pts > vol_a)).any():
        bad = int((((pts < 0) | (pts > vol_a)).any(dim=1)).sum())
        raise ValueError(f"{bad} particle position(s) outside the tomogram volume {vol_a.tolist()} A")
    if point_names is not None:
        _validate_point_names(point_names, p_count)

    # Project the FULL set once; split rows afterwards.
    projected, valid = model.project_volume(pts)
    projected_global, _ = model.project_volume_global(pts)
    weights_all = boundary_ramp_weights(projected, image_dims_a)
    disp_all, disp_flag, ctf_all, ctf_flag = _source_intermediates(model, pts)

    if heldout_fraction == 0:
        n_held = 0  # explicit no-split request (e.g. provenance IR of an exact conversion)
    elif p_count < 2 * min_heldout:
        n_held = 0
        warnings.warn(
            f"only {p_count} particles: no meaningful held-out split possible; "
            "held-out gates will report not_evaluated",
            stacklevel=2,
        )
    else:
        n_held = min(max(min_heldout, round(p_count * heldout_fraction)), p_count // 3)
    gen = torch.Generator().manual_seed(heldout_seed)
    perm = torch.randperm(p_count, generator=gen)
    held_idx, train_idx = perm[:n_held].sort().values, perm[n_held:].sort().values

    def names_at(idx):
        return [point_names[int(i)] for i in idx] if point_names is not None else None

    def split(arr, idx):
        return None if arr is None else arr[:, idx]

    ir = IRTiltSeries(
        native_model=model,
        grid_shape=None,
        points=pts[train_idx],
        source_projected=projected[:, train_idx].to(torch.float32),
        source_projected_global=projected_global[:, train_idx].to(torch.float32),
        sample_valid=torch.ones(train_idx.shape[0], dtype=torch.bool),
        projection_valid=valid[:, train_idx],
        weights=weights_all[:, train_idx],
        heldout_points=pts[held_idx],
        heldout_source_projected=projected[:, held_idx].to(torch.float32),
        heldout_projection_valid=valid[:, held_idx],
        heldout_weights=weights_all[:, held_idx],
        heldout_source_projected_global=projected_global[:, held_idx].to(torch.float32),
        meta=meta.model_copy(update={"displacement_3d": disp_flag, "source_ctf_depth": ctf_flag}),
        point_names=names_at(train_idx),
        heldout_point_names=names_at(held_idx),
        point_index=train_idx.to(torch.int64),
        heldout_point_index=held_idx.to(torch.int64),
        source_displacement_3d=split(disp_all, train_idx),
        heldout_source_displacement_3d=split(disp_all, held_idx),
        source_ctf_depth_a=split(ctf_all, train_idx),
        heldout_source_ctf_depth_a=split(ctf_all, held_idx),
    )
    ir.validate_optional_arrays()
    return ir


def build_ir_frame_series(
    model: FrameMotionModel,
    image_dims_a: torch.Tensor,
    *,
    meta: IRMeta,
    grid_shape: tuple[int, int] = (11, 11),
    heldout_seed: int = DEFAULT_HELDOUT_SEED,
) -> IRFrameSeries:
    """Map a regular 2D grid + Sobol held-out set through the source motion
    model (corrected/reference frame -> per-raw-frame positions)."""
    points = image_grid(image_dims_a, grid_shape)
    n_heldout = max(64, int(points.shape[0] * DEFAULT_HELDOUT_FACTOR))
    held = heldout_points(image_dims_a, n_heldout, seed=heldout_seed)

    mapped, valid = model.map_image(points)
    mapped_global, _ = model.map_image_global(points)
    held_mapped, held_valid = model.map_image(held)
    held_mapped_global, _ = model.map_image_global(held)

    weights = boundary_ramp_weights(mapped, image_dims_a)
    held_weights = boundary_ramp_weights(held_mapped, image_dims_a)

    return IRFrameSeries(
        native_model=model,
        grid_shape=grid_shape,
        points=points,
        source_projected=mapped.to(torch.float32),
        source_projected_global=mapped_global.to(torch.float32),
        sample_valid=torch.ones(points.shape[0], dtype=torch.bool),
        projection_valid=valid,
        weights=weights,
        heldout_points=held,
        heldout_source_projected=held_mapped.to(torch.float32),
        heldout_projection_valid=held_valid,
        heldout_weights=held_weights,
        heldout_source_projected_global=held_mapped_global.to(torch.float32),
        meta=meta,
    )
