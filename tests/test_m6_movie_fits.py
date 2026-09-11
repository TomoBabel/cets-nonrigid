"""M6: frame-series fits and .mcaln round trips (synthetic Warp movie with
populated movement + local grids and two pyramid levels)."""

import json
from pathlib import Path

import pytest
import torch

from cets_nonrigid.fit.movie_fits import fit_mcaln_shifts, fit_warp_movie
from cets_nonrigid.io.motion_txt import McAln, McAlnFrame
from cets_nonrigid.ir.build import build_ir_frame_series
from cets_nonrigid.ir.core import IRMeta
from cets_nonrigid.models.warp_movie import WarpMovieModel

GOLDEN_DIR = Path(__file__).parent / "golden"
PIX = 1.0  # alignment px = 1 A for the synthetic movie (dims 4000 x 3600)
FRAME_PX = (4000, 3600)


@pytest.fixture(scope="module")
def warp_movie():
    p = GOLDEN_DIR / "movie_synthetic.xml"
    if not p.exists():
        pytest.skip("synthetic movie fixture missing")
    with open(GOLDEN_DIR / "movie_positions.json") as f:
        golden = json.load(f)
    return WarpMovieModel.from_xml(
        p,
        n_frames=golden["n_frames"],
        image_dims_a=tuple(golden["image_dims"]),
        fraction_frames=golden["fraction_frames"],
    )


def _meta(f_count):
    return IRMeta(
        kind="frame_series",
        series_name="movie_synth",
        pixel_size_image_a=PIX,
        image_dims_px=FRAME_PX,
        projection_index=list(range(f_count)),
        projection_valid=[True] * f_count,
        projection_order=list(range(f_count)),
        projection_dose=[3.0 * i for i in range(f_count)],
        source_tool="warp-movie",
    )


@pytest.fixture(scope="module")
def warp_ir(warp_movie):
    return build_ir_frame_series(
        warp_movie, warp_movie.image_dims_a, meta=_meta(warp_movie.n_frames)
    )


def _mcaln_from_fit(fit, warp_movie):
    f_count = warp_movie.n_frames
    m = fit.model
    p_count = m.patch_centers_px.shape[0]
    return McAln(
        raw_frame_count=f_count * 15,
        integrated_frame_count=f_count,
        aligned_frame_count=f_count,
        alignment_image_size_px=FRAME_PX,
        alignment_pixel_size_a=PIX,
        patches=tuple(fit.meta["patch_grid"]),
        fm_ref=fit.meta["fm_ref"],
        frames=[
            McAlnFrame(
                integrated_index=i, source_start=i * 15, source_count=15,
                included=True, aligned_index=i,
            )
            for i in range(f_count)
        ],
        global_shifts=[
            (f, float(m.global_shifts_px[f, 0]), float(m.global_shifts_px[f, 1]))
            for f in range(f_count)
        ],
        local_shifts=[
            (
                p,
                [
                    (
                        f,
                        float(m.patch_centers_px[p, 0]),
                        float(m.patch_centers_px[p, 1]),
                        float(m.patch_shifts_px[f, p, 0]),
                        float(m.patch_shifts_px[f, p, 1]),
                        bool(m.patch_valid[f, p]),
                    )
                    for f in range(f_count)
                ],
            )
            for p in range(p_count)
        ],
    )


def test_w2m_fit_and_mcaln_roundtrip(warp_movie, warp_ir, tmp_path):
    fit = fit_mcaln_shifts(
        warp_ir, frame_size_px=FRAME_PX, pixel_size_a=PIX, patch_grid=(5, 5)
    )
    # Warp field magnitudes here are tens of A; the IDW representation cost
    # stays small.
    assert fit.rms_a_heldout < 3.0, fit.rms_a_heldout
    assert fit.coverage_heldout > 0.5

    mcaln = _mcaln_from_fit(fit, warp_movie)
    path = tmp_path / "movie.mcaln"
    mcaln.to_file(path)
    loaded = McAln.from_file(path)
    assert loaded.aligned_frame_count == warp_movie.n_frames
    assert loaded.patches == (5, 5)

    # The reloaded model reproduces the fitted model exactly.
    m0, m1 = fit.model, loaded.to_model()
    pts = torch.rand(30, 2, dtype=torch.float64) * torch.tensor(FRAME_PX, dtype=torch.float64)
    a, _ = m0.map_image(pts)
    b, _ = m1.map_image(pts)
    torch.testing.assert_close(a, b, rtol=0, atol=5e-3)  # %.3f px file rounding


def test_m2w_fit(warp_movie, warp_ir, tmp_path):
    """Round trip: Warp -> .mcaln model -> Warp movie grids."""
    w2m = fit_mcaln_shifts(
        warp_ir, frame_size_px=FRAME_PX, pixel_size_a=PIX, patch_grid=(5, 5)
    )

    ir_back = build_ir_frame_series(
        w2m.model,
        torch.tensor([FRAME_PX[0] * PIX, FRAME_PX[1] * PIX]),
        meta=_meta(warp_movie.n_frames),
    )
    from warpylib.movie import Movie

    m2w = fit_warp_movie(
        ir_back,
        Movie(),
        n_frames=warp_movie.n_frames,
        image_dims_a=(FRAME_PX[0] * PIX, FRAME_PX[1] * PIX),
        fraction_frames=1.0,
        local_grid=(3, 3, 4),
    )
    assert m2w.rms_a_heldout < 3.0, m2w.rms_a_heldout

    # Full round trip Warp -> mcaln -> Warp stays bounded.
    pts = torch.rand(50, 2, dtype=torch.float64) * torch.tensor(FRAME_PX, dtype=torch.float64)
    src, sv = warp_movie.map_image(pts)
    back, bv = m2w.model.map_image(pts)
    diff = (src.to(torch.float64) - back.to(torch.float64)).norm(dim=-1)
    # fm_ref gauge: compare modulo the per-series constant.
    sel = sv & bv
    diff[sel]
    const = (src.to(torch.float64) - back.to(torch.float64))[sel].mean(dim=0)
    d2 = ((src.to(torch.float64) - back.to(torch.float64)) - const).norm(dim=-1)[sel]
    assert float(d2.pow(2).mean().sqrt()) < 5.0, float(d2.pow(2).mean().sqrt())


def test_mcaln_validation_rejects_bad_files(tmp_path):
    text = (GOLDEN_DIR.parent / "golden").exists()  # noqa: F841
    good = McAln(
        raw_frame_count=30,
        integrated_frame_count=2,
        aligned_frame_count=2,
        alignment_image_size_px=(100, 100),
        alignment_pixel_size_a=1.0,
        patches=(1, 1),
        fm_ref=0,
        frames=[
            McAlnFrame(integrated_index=0, source_start=0, source_count=15,
                       included=True, aligned_index=0),
            McAlnFrame(integrated_index=1, source_start=15, source_count=15,
                       included=True, aligned_index=1),
        ],
        global_shifts=[(0, 0.0, 0.0), (1, 1.0, -1.0)],
        local_shifts=[(0, [(0, 50.0, 50.0, 0.1, 0.2, True), (1, 50.0, 50.0, 0.3, 0.4, True)])],
    )
    s = good.to_string()
    assert McAln.from_string(s).fm_ref == 0

    with pytest.raises(ValueError, match="magic"):
        McAln.from_string("garbage\n" + s)
    with pytest.raises(ValueError, match="globalShift"):
        bad = good.model_copy(deep=True)
        bad.global_shifts = [(0, 0.0, 0.0), (0, 1.0, 1.0)]  # duplicate index
        bad.to_string()
    with pytest.raises(ValueError, match="non-finite"):
        bad = good.model_copy(deep=True)
        bad.global_shifts = [(0, 0.0, 0.0), (1, float("nan"), 0.0)]
        bad.to_string()


def test_w2m_m2w_pipelines(tmp_path, warp_movie):
    """End-to-end pipeline functions (CLI backends) on the synthetic movie."""

    from cets_nonrigid.convert import mcaln_to_warp_movie, warp_movie_to_mcaln
    from cets_nonrigid.io.warp_xml import load_warp_tiltseries  # noqa: F401

    golden_xml = GOLDEN_DIR / "movie_synthetic.xml"
    out_mcaln = tmp_path / "conv.mcaln"
    r1 = warp_movie_to_mcaln(
        golden_xml,
        out_mcaln,
        image_size_px=FRAME_PX,
        pixel_size_a=PIX,
        n_frames=warp_movie.n_frames,
        fraction_frames=warp_movie.fraction_frames,
        raw_frames_per_aligned=15,
    )
    assert r1.fit.rms_a_heldout < 3.0
    assert r1.fit.rms_a_heldout is not None

    out_xml = tmp_path / "back.xml"
    r2 = mcaln_to_warp_movie(out_mcaln, golden_xml, out_xml)
    assert r2.fit.rms_a_heldout < 3.0
    # Template CTF grid preserved, pyramids stripped.
    from lxml import etree

    root = etree.fromstring(out_xml.read_bytes())
    assert root.find("GridCTF") is not None
    assert root.find("PyramidShiftX") is None
    assert root.find("GridLocalMovementX").get("Width") == "3"
