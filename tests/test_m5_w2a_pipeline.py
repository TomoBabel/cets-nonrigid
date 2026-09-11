"""M5: w2a end-to-end + both round trips on real data (Position_16_3)."""

from pathlib import Path

import pytest
import torch
from lxml import etree

from cets_nonrigid.convert import aretomo_to_warp, warp_to_aretomo
from cets_nonrigid.io.aln import load_aln, match_tilts
from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.ir.sampling import heldout_points

WARP_TRIAL = Path("/hpc/projects/group.czii/utz.ermel/warp_trial/24jul16a")
AT3_ALN = Path(
    "/hpc/projects/group.czii/utz.ermel/repos/arewarpo/testdata_runs/at3_24jul16a/out/"
    "24jul16a_Position_16_3.aln"
)
SERIES = "24jul16a_Position_16_3"
PIX = 1.54
VOL_PX = (4096, 4096, 2000)
VOL_A = tuple(d * PIX for d in VOL_PX)


@pytest.fixture
def template_xml(tmp_path):
    xml = WARP_TRIAL / "warp_tiltseries" / f"{SERIES}.xml"
    if not xml.exists():
        pytest.skip("warp_trial not available")
    root = etree.fromstring(xml.read_bytes())
    root.set("ImageDimensionsAngstrom", f"{4096 * PIX}, {4096 * PIX}")
    root.set("VolumeDimensionsAngstrom", f"{VOL_A[0]}, {VOL_A[1]}, {VOL_A[2]}")
    p = tmp_path / f"{SERIES}.xml"
    p.write_bytes(etree.tostring(root, xml_declaration=True, encoding="utf-8"))
    return p


def _fidelity_rms_px(warp_model, aln_series, n=400, seed=123):
    """RMS between the two FULL models on an independent point set."""
    match = match_tilts(
        warp_model.ts.angles, torch.ones(warp_model.ts.n_tilts), aln_series
    )
    pts = heldout_points(warp_model.model.volume_dims_a, n, seed=seed)
    w_xy, w_valid = warp_model.model.project_volume(pts)
    a_xy, _ = aln_series.model.project_volume(pts)
    perm = torch.tensor(match.aln_to_warp)
    sel = w_valid[perm]
    d = (w_xy[perm].to(torch.float64) - a_xy.to(torch.float64)).norm(dim=-1)[sel]
    return float(d.pow(2).mean().sqrt()) / PIX


def test_w2a_end_to_end(template_xml, tmp_path):
    out_aln = tmp_path / "converted.aln"

    result = warp_to_aretomo(
        template_xml, out_aln, pixel_size_a=PIX
    )

    # Zero level angles in this XML: globals are exactly representable.
    assert result.global_fit.rms_px_heldout < 0.05, result.global_fit.rms_px_heldout
    # Local field (Warp movement splines -> IDW patches) is lossy but small.
    assert result.local_fit.rms_px_heldout < 5.0, result.local_fit.rms_px_heldout
    assert result.local_fit.coverage_heldout > 0.3
    assert set(result.local_fit.z_stratified_rms_px) == {"z_low", "z_mid", "z_high"}

    # Emitted .aln parses through the strict adapter, with Thickness header.
    aln_series = load_aln(out_aln, pixel_size_a=PIX, volume_dims_a=VOL_A)
    assert aln_series.aln.NumPatches == 25
    assert aln_series.aln.Thickness == VOL_PX[2]
    assert len(aln_series.aln.LocalAlignments) == 25 * len(aln_series.aln.GlobalAlignments)

    # Conversion fidelity: the emitted .aln model reproduces the source Warp
    # model on independent points.
    warp_series = load_warp_tiltseries(template_xml)
    rms = _fidelity_rms_px(warp_series, aln_series)
    assert rms < 5.0, rms

    # TILT rows ascend (AreTomo convention).
    tilts = [g.tilt for g in aln_series.aln.GlobalAlignments]
    assert tilts == sorted(tilts)


def test_round_trip_w2a2w(template_xml, tmp_path):
    """Warp -> .aln -> Warp: projections must survive both lossy steps."""
    mid_aln = tmp_path / "mid.aln"
    warp_to_aretomo(template_xml, mid_aln, pixel_size_a=PIX)

    back_xml = tmp_path / "back.xml"
    aretomo_to_warp(mid_aln, template_xml, back_xml, movement_grid=(6, 6))

    src = load_warp_tiltseries(template_xml)
    back = load_warp_tiltseries(back_xml)

    pts = heldout_points(src.model.volume_dims_a, 400, seed=7)
    s_xy, s_valid = src.model.project_volume(pts)
    b_xy, b_valid = back.model.project_volume(pts)
    sel = s_valid & b_valid
    d = (s_xy.to(torch.float64) - b_xy.to(torch.float64)).norm(dim=-1)[sel]
    rms_px = float(d.pow(2).mean().sqrt()) / PIX
    assert rms_px < 6.0, rms_px


@pytest.mark.skipif(not AT3_ALN.exists(), reason="genuine AreTomo3 aln missing")
def test_round_trip_a2w2a(template_xml, tmp_path):
    """.aln -> Warp -> .aln: compare the two .aln models as FIELDS."""
    mid_xml = tmp_path / "mid.xml"
    aretomo_to_warp(AT3_ALN, template_xml, mid_xml, movement_grid=(6, 6))

    back_aln = tmp_path / "back.aln"
    warp_to_aretomo(mid_xml, back_aln, pixel_size_a=PIX, patch_grid=(4, 4))

    src = load_aln(AT3_ALN, pixel_size_a=PIX, volume_dims_a=VOL_A)
    back = load_aln(back_aln, pixel_size_a=PIX, volume_dims_a=VOL_A)

    pts = heldout_points(torch.tensor(VOL_A), 400, seed=11)
    s_xy, s_valid = src.model.project_volume(pts)
    # Row orders may differ; match via tilt angles.
    s_stage = src.stage_angles_deg
    b_stage = back.stage_angles_deg
    perm = [int(torch.argmin((b_stage - a).abs())) for a in s_stage]
    b_xy, _ = back.model.project_volume(pts)
    d = (s_xy.to(torch.float64) - b_xy[perm].to(torch.float64)).norm(dim=-1)[s_valid]
    rms_px = float(d.pow(2).mean().sqrt()) / PIX
    assert rms_px < 6.0, rms_px
