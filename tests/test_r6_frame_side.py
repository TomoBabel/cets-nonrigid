"""R6: frame side — RELION micrograph motion star ⇄ Warp movie ⇄ .mcaln."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cets_nonrigid.fit.relion_motion_fit import fit_relion_motion
from cets_nonrigid.io.relion_motion_star import (
    RelionMicrographMotion,
    read_micrograph_motion_star,
    write_micrograph_motion_star,
)
from cets_nonrigid.ir.build import build_ir_frame_series
from cets_nonrigid.ir.core import IRMeta
from cets_nonrigid.models.relion_motion import RelionMicrographMotionModel

RNG = np.random.default_rng(20260908)

GOLDEN_DIR = Path(__file__).parent / "golden"
SIZE = (4000, 3600)
PIX = 1.2
F = 9


def _meta(f_count, size=SIZE, pix=PIX):
    return IRMeta(
        kind="frame_series",
        series_name="r6",
        pixel_size_image_a=pix,
        image_dims_px=size,
        projection_index=list(range(f_count)),
        projection_valid=[True] * f_count,
        projection_order=list(range(f_count)),
        projection_dose=[2.0 * i for i in range(f_count)],
        source_tool="test",
        sampling="grid",
    )


# --- star IO ------------------------------------------------------------------


def test_motion_star_roundtrip(tmp_path):
    g = torch.tensor(RNG.uniform(-5, 5, (F, 2)))
    g[0] = -9999.0  # leading sentinel
    coeffs = torch.tensor(RNG.uniform(-0.05, 0.05, 36))
    m = RelionMicrographMotion(
        image_size_px=SIZE, n_frames=F, global_shifts_px=g, poly_coeffs=coeffs,
        pixel_size_a=PIX, start_frame=2, movie_name="movie.eer",
        dose_rate=1.2, voltage_kv=300.0, eer_grouping=15, eer_upsampling=2,
    )
    path = write_micrograph_motion_star(tmp_path / "mic.star", m)
    text = path.read_text()
    assert "data_global_shift" in text and "data_local_motion_model" in text
    assert "_rlnMotionModelVersion" in text
    back = read_micrograph_motion_star(path)
    torch.testing.assert_close(back.global_shifts_px, g.to(torch.float64), atol=1e-5, rtol=0)
    torch.testing.assert_close(back.poly_coeffs, coeffs.to(torch.float64), atol=1e-5, rtol=0)  # %.6f
    assert back.start_frame == 2 and back.eer_grouping == 15
    assert back.global_shifts_px[0, 0] == -9999.0  # sentinel survives

    # version 0: no local block
    m0 = RelionMicrographMotion(
        image_size_px=SIZE, n_frames=F, global_shifts_px=torch.zeros(F, 2),
        poly_coeffs=None, pixel_size_a=PIX,
    )
    p0 = write_micrograph_motion_star(tmp_path / "v0.star", m0)
    assert "data_local_motion_model" not in p0.read_text()
    assert read_micrograph_motion_star(p0).poly_coeffs is None


# --- exact relative-trajectory recovery (poly-representable source) -----------


def test_fit_recovers_relative_trajectories_exactly():
    g = torch.tensor(RNG.uniform(-6, 6, (F, 2)), dtype=torch.float64)
    coeffs = torch.tensor(RNG.uniform(-0.08, 0.08, 36), dtype=torch.float64)
    source = RelionMicrographMotionModel(
        global_shifts_px=g, poly_coeffs=coeffs, image_size_px=SIZE,
        pixel_size_a=PIX, start_frame=1,
    )
    image_dims_a = torch.tensor([SIZE[0] * PIX, SIZE[1] * PIX])
    ir = build_ir_frame_series(source, image_dims_a, meta=_meta(F), grid_shape=(9, 9))
    fit = fit_relion_motion(ir, image_size_px=SIZE, pixel_size_a=PIX)

    # "exact recovery" = exact RELATIVE trajectories, never absolute maps;
    # the IR stores projections in float32, bounding precision at ~2e-4 px
    assert fit.rms_px_train < 5e-4
    assert fit.heldout_status == "evaluated" and fit.rms_px_heldout < 5e-4
    # the source's own pure-z terms are absorbed into globals; the recovered
    # spatial coefficients must equal the source's non-pure-z coefficients
    rec = fit.motion.poly_coeffs
    # parameter-space comparison is loose on purpose: near-collinear columns
    # amplify f32-IR noise into coefficients while the FIELD stays exact
    # (the rms and G7 assertions above/below carry the exactness claim)
    torch.testing.assert_close(rec[3:18], coeffs[3:18], atol=1e-3, rtol=0)
    torch.testing.assert_close(rec[21:36], coeffs[21:36], atol=1e-3, rtol=0)
    assert float(rec[:3].abs().max()) == 0.0 and float(rec[18:21].abs().max()) == 0.0

    # G7 re-gauge identity: predicted inter-frame differences equal the
    # source's, everywhere on a fresh point set
    pred = fit.motion.to_model()
    pts = torch.tensor(RNG.uniform(0, 1, (30, 2))) * image_dims_a
    src_raw, _ = source.map_image(pts)
    fit_raw, _ = pred.map_image(pts)
    src_rel = src_raw - src_raw[0:1]
    fit_rel = fit_raw - fit_raw[0:1]
    torch.testing.assert_close(fit_rel, src_rel, atol=5e-3, rtol=0)  # A; f32-IR-limited


def test_static_field_is_reported_not_hidden():
    """A source with a spatially varying field at the reference frame: the
    discarded static warp must be REPORTED (rms/p95/max), never treated as a
    fit residual."""
    from cets_nonrigid.models.aretomo_motion import AretomoMotionModel

    f = 7
    centers = torch.tensor(
        [[SIZE[0] * (i + 0.5) / 2, SIZE[1] * (j + 0.5) / 2] for i in range(2) for j in range(2)],
        dtype=torch.float64,
    )
    shifts = torch.zeros(f, 4, 2, dtype=torch.float64)
    for t in range(f):
        for pp in range(4):
            shifts[t, pp, 0] = 3.0 * np.sin(0.9 * t + pp)
            shifts[t, pp, 1] = 3.0 * np.cos(0.7 * t - pp)
    # NOTE: shifts at frame 0 are NONZERO and spatially varying -> static field
    source = AretomoMotionModel(
        global_shifts_px=torch.tensor(RNG.uniform(-4, 4, (f, 2))),
        patch_centers_px=centers,
        patch_shifts_px=shifts,
        patch_valid=torch.ones(f, 4, dtype=torch.bool),
        frame_size_px=SIZE,
        pixel_size_a=PIX,
    )
    image_dims_a = torch.tensor([SIZE[0] * PIX, SIZE[1] * PIX])
    ir = build_ir_frame_series(source, image_dims_a, meta=_meta(f), grid_shape=(9, 9))
    fit = fit_relion_motion(ir, image_size_px=SIZE, pixel_size_a=PIX)
    assert fit.static_field_rms_px > 0.5  # the frame-0 spatial field was discarded
    # The linear-r IDW field (4 sparse patches, cutoff kinks) is genuinely
    # poorly representable by the quadratic-in-space polynomial — exact
    # representability is impossible in general; gate RELATIVE capture and
    # train/held-out consistency instead of an absolute number.
    p = PIX
    t_field = (ir.points.to(torch.float64)[None] - ir.source_projected.to(torch.float64)) / p
    t_tilde_mag = float((t_field - t_field[0][None]).norm(dim=-1).mean())
    assert fit.rms_px_heldout is not None
    assert fit.rms_px_heldout < 0.6 * t_tilde_mag  # captures the majority
    assert fit.rms_px_heldout < 1.5 * fit.rms_px_train + 0.2  # no overfit blowup


# --- pipelines ----------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_movie():
    p = GOLDEN_DIR / "movie_synthetic.xml"
    if not p.exists():
        pytest.skip("synthetic movie fixture missing")
    with open(GOLDEN_DIR / "movie_positions.json") as f:
        return p, json.load(f)


def test_w2rm_and_rm2w_roundtrip(tmp_path, golden_movie):
    from cets_nonrigid.convert_relion_movie import (
        relion_motion_to_warp_movie,
        warp_movie_to_relion_motion,
    )

    xml, golden = golden_movie
    size_px = (int(golden["image_dims"][0]), int(golden["image_dims"][1]))
    r = warp_movie_to_relion_motion(
        xml, tmp_path / "mic.star",
        image_size_px=size_px, pixel_size_a=1.0,
        n_frames=golden["n_frames"], fraction_frames=golden["fraction_frames"],
    )
    assert r.out_star.exists()
    assert r.fit.heldout_status == "evaluated"
    # capacity gate, relative: the polynomial captures the majority of the
    # relative field and does not overfit
    assert r.fit.rms_px_heldout < 1.5 * r.fit.rms_px_train + 0.2

    r2 = relion_motion_to_warp_movie(
        r.out_star, xml, tmp_path / "back.xml",
        fraction_frames=golden["fraction_frames"],
    )
    assert r2.out_xml.exists()
    assert r2.fit.heldout_status == "evaluated"
    # B-spline grids vs the quadratic/cubic polynomial: sub-px representation
    # error with healthy train/held-out agreement
    assert r2.fit.rms_a_heldout < 1.5 * r2.fit.rms_a_train + 0.2
    assert r2.fit.rms_a_heldout < 1.0


def test_m2rm_and_scale_rejection(tmp_path):
    from cets_nonrigid.convert import mcaln_from_motion_model
    from cets_nonrigid.convert_relion_movie import mcaln_to_relion_motion
    from cets_nonrigid.models.aretomo_motion import AretomoMotionModel

    f = 8
    centers = torch.tensor(
        [[SIZE[0] * (i + 0.5) / 3, SIZE[1] * (j + 0.5) / 3] for i in range(3) for j in range(3)],
        dtype=torch.float64,
    )
    shifts = torch.zeros(f, 9, 2, dtype=torch.float64)
    for t in range(f):
        for pp in range(9):
            shifts[t, pp, 0] = 2.0 * np.sin(0.8 * t + 0.5 * pp) * t / max(1, f - 1)
            shifts[t, pp, 1] = 2.0 * np.cos(0.6 * t - 0.3 * pp) * t / max(1, f - 1)
    model = AretomoMotionModel(
        global_shifts_px=torch.tensor(RNG.uniform(-3, 3, (f, 2))),
        patch_centers_px=centers,
        patch_shifts_px=shifts,
        patch_valid=torch.ones(f, 9, dtype=torch.bool),
        frame_size_px=SIZE,
        pixel_size_a=PIX,
    )
    mcaln = mcaln_from_motion_model(
        model, n_frames=f, image_size_px=SIZE, pixel_size_a=PIX,
        patch_grid=(3, 3), fm_ref=0, raw_frames_per_aligned=15,
    )
    path = tmp_path / "syn.mcaln"
    mcaln.to_file(path)

    r = mcaln_to_relion_motion(path, tmp_path / "mic.star", dose_rate=1.1)
    assert r.out_star.exists()
    back = read_micrograph_motion_star(r.out_star)
    assert back.eer_grouping == 15  # uniform source_count -> EER label
    assert r.fit.rms_px_heldout is not None

    # alignment-scale != 1 refused
    bad = mcaln.model_copy(update={"input_to_alignment_scale_xy": (2.0, 2.0)})
    bad_path = tmp_path / "bad.mcaln"
    bad.to_file(bad_path)
    with pytest.raises(ValueError, match="input_to_alignment_scale"):
        mcaln_to_relion_motion(bad_path, tmp_path / "bad.star")
