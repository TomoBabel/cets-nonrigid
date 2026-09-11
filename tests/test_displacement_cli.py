"""CETS CLI volume-warp fitting and persisted diagnostic reports."""

import json
from pathlib import Path
import pytest
from click.testing import CliRunner
from cets_nonrigid import api
from cets_nonrigid.cli import main
from cets_nonrigid.cli.exchange import Grid
from cets_nonrigid.io.warp_xml import load_warp_tiltseries

GOLDEN_XML = Path("tests/golden/TS_1_volwarp.xml")


def _golden_store(tmp_path, grid=(15, 15, 5)):
    path = tmp_path / "source.cets.json"
    api.to_cets("warp", GOLDEN_XML, pixel_size_a=0.834, grid_shape=grid, output=path)
    return path


def test_volume_warp_grid_option_parsing():
    grid = Grid(4)
    assert grid.convert("3x3x2x4", None, None) == (3, 3, 2, 4)
    assert grid.convert("3X3x2xT", None, None) == (3, 3, 2, None)
    for bad in ("3x3x2", "3x3x2x0", "ax3x2x4", "3x3x2xq"):
        with pytest.raises(Exception):
            grid.convert(bad, None, None)


def test_volume_warp_fit_and_report_bundle(tmp_path):
    source = _golden_store(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "fit",
            "warp",
            str(source),
            "-o",
            str(tmp_path / "fit.xml"),
            "--template-xml",
            str(GOLDEN_XML),
            "--global-mode",
            "template",
            "--movement-grid",
            "6x4",
            "--volume-warp-grid",
            "3x3x2x4",
            "--report-bundle",
            str(tmp_path / "fit.cets.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    metrics = json.loads(result.output)
    assert metrics["volume_warp_fitted"] and metrics["volume_warp_grid"] == [3, 3, 2, 4]
    assert metrics["volume_warp_data_rank"] == 72
    assert metrics["volume_warp_rms_a_heldout"] < 1e-3
    assert metrics["rms_a_heldout"] < 1e-2
    assert tuple(load_warp_tiltseries(tmp_path / "fit.xml").ts.grid_volume_warp_x.dimensions) == (3, 3, 2, 4)
    bundle = api.read_bundle(tmp_path / "fit.cets.json")
    assert "volume_warp_fitted" in json.dumps(bundle.reports)
    assert api.read_bundle(source).reports != bundle.reports


def test_movement_only_and_temporal_grid(tmp_path):
    source = _golden_store(tmp_path, grid=(8, 8, 4))
    bundle = api.read_bundle(source)
    default = api.fit(bundle, "warp", template_xml=GOLDEN_XML, global_mode="template")
    assert default.metrics["volume_warp_fitted"] is False
    fitted = api.fit(bundle, "warp", template_xml=GOLDEN_XML, global_mode="template", volume_warp_grid=(2, 2, 2, None))
    assert fitted.metrics["volume_warp_grid"] == [2, 2, 2, 41]
    assert fitted.metrics["volume_warp_data_rank"] == 328


def test_zero_displacement_refuses_volume_fit(tmp_path):
    from test_ir_cli import _warp_store

    _, path, _ = _warp_store(tmp_path)
    with pytest.raises(ValueError, match="available 3D displacement"):
        api.fit(path, "warp", volume_warp_grid=(2, 2, 1, 4))
    assert not api.fit(path, "warp").metrics["volume_warp_fitted"]
