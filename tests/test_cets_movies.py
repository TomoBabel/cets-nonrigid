"""All six movie exchanges, temporal identity and scientific metadata."""

import itertools
import pytest
import torch
from cets_nonrigid import api
from cets_nonrigid.io.relion_motion_star import (
    RelionMicrographMotion,
    write_micrograph_motion_star,
    read_micrograph_motion_star,
)

FORMATS = ("warp-movie", "mcaln", "relion-motion")


@pytest.fixture
def movie_sources(tmp_path):
    n = 12
    time = torch.arange(n, dtype=torch.float64)
    coeffs = torch.zeros(36, dtype=torch.float64)
    coeffs[3] = 0.003
    coeffs[21] = -0.002
    motion = RelionMicrographMotion(
        image_size_px=(96, 80),
        n_frames=n,
        global_shifts_px=torch.stack((time * 0.3, time * -0.1), dim=1),
        poly_coeffs=coeffs,
        pixel_size_a=2.0,
        movie_name="movie.eer",
        dose_rate=0.5,
        pre_exposure=2.0,
        voltage_kv=300.0,
        eer_grouping=10,
    )
    path = tmp_path / "source.star"
    write_micrograph_motion_star(path, motion)
    source = api.to_cets("relion-motion", path, grid_shape=(7, 7))
    bundles = {"relion-motion": source}
    for target, suffix in [("warp-movie", ".xml"), ("mcaln", ".mcaln")]:
        result = api.fit(source, target)
        output = tmp_path / (target + suffix)
        api.export_native(result, output)
        options = {"image_size_px": (96, 80), "pixel_size_a": 2.0, "n_frames": n} if target == "warp-movie" else {}
        bundles[target] = api.to_cets(target, output, grid_shape=(7, 7), **options)
    return bundles


@pytest.mark.parametrize("source,target", list(itertools.permutations(FORMATS, 2)))
def test_each_movie_direction(movie_sources, source, target, tmp_path):
    bundle = movie_sources[source]
    path = tmp_path / (source + ".cets.json")
    api.write_bundle(bundle, path)
    loaded = api.read_bundle(path)
    result = api.fit(loaded, target)
    assert result.files
    metric = result.metrics.get("rms_a_heldout", result.metrics.get("rms_px_heldout"))
    assert metric is not None and metric < 0.05
    assert result.metrics["heldout_status"] == "evaluated"


def test_relion_movie_metadata_roundtrip(movie_sources, tmp_path):
    bundle = movie_sources["relion-motion"]
    assert bundle.context.rows[4].accumulated_dose == 4.0
    assert bundle.context.rows[4].exposure_dose == 0.5
    result = api.fit(bundle, "relion-motion")
    out = tmp_path / "returned.star"
    api.export_native(result, out)
    motion = read_micrograph_motion_star(out)
    assert (motion.voltage_kv, motion.dose_rate, motion.pre_exposure, motion.eer_grouping) == (300.0, 0.5, 2.0, 10)
    assert motion.movie_name == "movie.eer"


def test_missing_movie_rows_preserve_temporal_indices(tmp_path):
    n = 8
    shifts = torch.stack((torch.arange(n) * 0.2, torch.arange(n) * -0.1), dim=1).to(torch.float64)
    shifts[3] = -9999
    motion = RelionMicrographMotion(
        image_size_px=(96, 80),
        n_frames=n,
        global_shifts_px=shifts,
        poly_coeffs=None,
        pixel_size_a=2.0,
        movie_name="test.eer",
    )
    path = tmp_path / "missing.star"
    write_micrograph_motion_star(path, motion)
    bundle = api.to_cets("relion-motion", path)
    assert len(bundle.context.row_ids) == n
    assert not bundle.context.operators()[2][3]
    assert not bundle.samples.training.observation_valid[3].any()
    assert not bundle.samples.training.projected_residual[3].any()
    out = tmp_path / "returned.star"
    api.export_native(api.fit(bundle, "relion-motion"), out)
    returned = read_micrograph_motion_star(out)
    assert returned.n_frames == n and returned.global_shifts_px[3, 0] == -9999
    mcaln = api.fit(bundle, "mcaln")
    out_mc = tmp_path / "returned.mcaln"
    api.export_native(mcaln, out_mc)
    from cets_nonrigid.io.motion_txt import McAln

    parsed = McAln.from_file(out_mc)
    assert parsed.integrated_frame_count == n and not parsed.frames[3].included
