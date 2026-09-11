"""M4: a2w end-to-end on real data (warp_trial Position_16_3).

Prefers the genuine AreTomo3 2.3.1 .aln produced under testdata_runs/ (run on
this node); falls back to the AreTomo2 file from warp_trial for the pipeline
mechanics (acceptable for testing per project decision).
"""

from pathlib import Path

import pytest
import torch
from lxml import etree

from cets_nonrigid.convert import aretomo_to_warp
from cets_nonrigid.io.warp_xml import load_warp_tiltseries

WARP_TRIAL = Path("/hpc/projects/group.czii/utz.ermel/warp_trial/24jul16a")
AT3_OUT = Path(
    "/hpc/projects/group.czii/utz.ermel/repos/arewarpo/testdata_runs/at3_24jul16a/out"
)
SERIES = "24jul16a_Position_16_3"
PIX = 1.54
VOL_PX = (4096, 4096, 2000)


def _aln_path():
    at3 = AT3_OUT / f"{SERIES}.aln"
    if at3.exists():
        return at3
    at2 = WARP_TRIAL / "portal_data" / SERIES / "Alignments" / "100" / f"{SERIES}.aln"
    if at2.exists():
        return at2
    pytest.skip("no .aln available for Position_16_3")


@pytest.fixture
def template_xml(tmp_path):
    xml = WARP_TRIAL / "warp_tiltseries" / f"{SERIES}.xml"
    if not xml.exists():
        pytest.skip("warp_trial not available")
    root = etree.fromstring(xml.read_bytes())
    root.set("ImageDimensionsAngstrom", f"{4096 * PIX}, {4096 * PIX}")
    root.set(
        "VolumeDimensionsAngstrom",
        f"{VOL_PX[0] * PIX}, {VOL_PX[1] * PIX}, {VOL_PX[2] * PIX}",
    )
    p = tmp_path / f"{SERIES}.xml"
    p.write_bytes(etree.tostring(root, xml_declaration=True, encoding="utf-8"))
    return p


def test_a2w_end_to_end(template_xml, tmp_path):
    aln = _aln_path()
    out_xml = tmp_path / "converted.xml"

    result = aretomo_to_warp(aln, template_xml, out_xml)

    # The closed-form global mapping is exact by construction.
    assert result.global_check_rms_a < 0.5, result.global_check_rms_a
    assert result.pixel_size_a == pytest.approx(PIX, abs=1e-3)

    # Local fit: IDW field into movement splines is lossy but small.
    fit = result.fit
    assert fit.rms_a_heldout < 15.0, fit.rms_a_heldout
    assert fit.coverage_heldout > 0.3

    # Output XML is modern-format and loads through the strict loader.
    converted = load_warp_tiltseries(out_xml)
    assert converted.ts.level_angle_x == 0.0 and converted.ts.level_angle_y == 0.0
    assert converted.ts.grid_movement_x.values.abs().max() > 0
    assert converted.ts.grid_volume_warp_x.dimensions == (1, 1, 1, 1)

    # Angles were rewritten to -(aln TILT) on matched tilts.
    t_aln0 = 0
    t_warp0 = result.match.aln_to_warp[t_aln0]
    from cets_nonrigid.io.aln import load_aln

    aln_series = load_aln(
        aln, pixel_size_a=PIX, volume_dims_a=tuple(d * PIX for d in VOL_PX)
    )
    assert float(converted.ts.angles[t_warp0]) == pytest.approx(
        -float(aln_series.model.tilt_deg[t_aln0]), abs=1e-3
    )

    # Non-template nodes survived untouched (CTF fit block from the template).
    src_root = etree.fromstring(template_xml.read_bytes())
    out_root = etree.fromstring(out_xml.read_bytes())
    for node in ("GridCTF", "CTF"):
        s, o = src_root.find(node), out_root.find(node)
        if s is not None:
            assert o is not None and etree.tostring(s) == etree.tostring(o)

    # CETS wire round trips live in test_cets_exchange; retain independent
    # held-out availability and all projection/CTF assertions in this real-data test.
    assert result.fit.heldout_status == "evaluated"
    assert result.fit.per_tilt_rms_a_heldout is not None


def test_a2w_converted_projections_match_source(template_xml, tmp_path):
    """The converted Warp model must reproduce the AreTomo model's projections
    on an independent point set (the actual conversion fidelity criterion)."""
    from cets_nonrigid.io.aln import load_aln, match_tilts
    from cets_nonrigid.ir.sampling import heldout_points

    aln = _aln_path()
    out_xml = tmp_path / "conv.xml"
    aretomo_to_warp(aln, template_xml, out_xml)

    converted = load_warp_tiltseries(out_xml)
    aln_series = load_aln(
        aln, pixel_size_a=PIX, volume_dims_a=tuple(d * PIX for d in VOL_PX)
    )
    match = match_tilts(converted.ts.angles, torch.ones(converted.n_tilts), aln_series)

    pts = heldout_points(converted.model.volume_dims_a, 400, seed=99)
    w_xy, w_valid = converted.model.project_volume(pts)
    a_xy, _ = aln_series.model.project_volume(pts)
    perm = torch.tensor(match.aln_to_warp)
    sel = w_valid[perm]
    diff = (w_xy[perm].to(torch.float64) - a_xy.to(torch.float64)).norm(dim=-1)[sel]
    rms_px = float(diff.pow(2).mean().sqrt()) / PIX
    assert rms_px < 10.0, rms_px
