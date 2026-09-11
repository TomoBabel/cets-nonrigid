"""M6: Warp movie model vs the C# golden (Movie.GetPositionInAllFrames)."""

import json
from pathlib import Path

import pytest
import torch

from cets_nonrigid.models.warp_movie import WarpMovieModel

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def golden():
    p = GOLDEN_DIR / "movie_positions.json"
    if not p.exists():
        pytest.skip("movie golden missing")
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model(golden):
    return WarpMovieModel.from_xml(
        GOLDEN_DIR / "movie_synthetic.xml",
        n_frames=golden["n_frames"],
        image_dims_a=tuple(golden["image_dims"]),
        fraction_frames=golden["fraction_frames"],
    )


def test_grids_loaded(model):
    assert len(model.movie.pyramid_shift_x) == 2
    assert model.movie.grid_local_x.values.abs().max() > 0
    assert model.movie.grid_movement_x.dimensions == (1, 1, 6)


def test_matches_csharp_golden(golden, model):
    points = torch.tensor(golden["points"], dtype=torch.float64)  # (N, 2)
    raw, _ = model.map_image(points)  # (F, N, 2)

    n = points.shape[0]
    f_count = golden["n_frames"]
    expected = torch.tensor(golden["positions"], dtype=torch.float32).reshape(n, f_count, 3)
    expected_xy = expected[..., :2].permute(1, 0, 2)  # (F, N, 2)

    # float32 chain over ~4000 A coordinates: agree to ~1e-3 A.
    torch.testing.assert_close(raw, expected_xy, rtol=1e-6, atol=2e-3)


def test_fraction_frames_matters(golden):
    m1 = WarpMovieModel.from_xml(
        GOLDEN_DIR / "movie_synthetic.xml",
        n_frames=golden["n_frames"],
        image_dims_a=tuple(golden["image_dims"]),
        fraction_frames=1.0,
    )
    points = torch.tensor(golden["points"], dtype=torch.float64)
    raw_frac, _ = WarpMovieModel.from_xml(
        GOLDEN_DIR / "movie_synthetic.xml",
        n_frames=golden["n_frames"],
        image_dims_a=tuple(golden["image_dims"]),
        fraction_frames=golden["fraction_frames"],
    ).map_image(points)
    raw_one, _ = m1.map_image(points)
    assert (raw_frac - raw_one).abs().max() > 0.05


def test_global_vs_full(golden, model):
    points = torch.tensor(golden["points"], dtype=torch.float64)
    full, _ = model.map_image(points)
    glob, _ = model.map_image_global(points)
    # The global-only SHIFT (input minus output) is spatially constant per
    # frame ((1,1,F) grid); the full shift is not.
    shift = points.to(torch.float32)[None] - glob
    per_frame_spread = (shift - shift[:, :1]).abs().max()
    assert per_frame_spread < 1e-3  # float32 rounding of (x - (x - s)) at ~4000 A
    assert (full - glob).abs().max() > 0.5
