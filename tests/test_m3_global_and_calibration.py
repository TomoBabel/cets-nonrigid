"""M3: strict Warp XML loading, w2a global fit, sign calibration.

The real-data calibration test uses the warp_trial dataset (the only
Warp-processed dataset at hand): Warp XMLs are old-format, so a dim-patched
COPY is created under tmp_path (the tool itself rejects old-format files).
The matching .aln files there are AreTomo2 output — acceptable for testing
per project decision (their global geometry is what calibration exercises);
genuine AreTomo3 runs replace them for final convention pinning.
"""

from pathlib import Path

import pytest
import torch
from lxml import etree

from cets_nonrigid.fit.aretomo_global import closed_form_init, fit_aretomo_globals
from cets_nonrigid.fit.calibrate import CONFIGURED, calibrate_signs
from cets_nonrigid.io.aln import load_aln, match_tilts
from cets_nonrigid.io.warp_xml import load_warp_tiltseries

WARP_TRIAL = Path("/hpc/projects/group.czii/utz.ermel/warp_trial/24jul16a")
TRIAL_SERIES = "24jul16a_Position_30_2"
TRIAL_PIX = 1.54
TRIAL_IMG_PX = (4096, 4096)
TRIAL_VOL_PX = (4096, 4096, 2000)


# ---------------------------------------------------------------------------
# Strict loading
# ---------------------------------------------------------------------------


def test_strict_loader_loads_modern_xml(ts1_xml_path):
    series = load_warp_tiltseries(ts1_xml_path)
    assert series.n_tilts == 41
    assert series.model.volume_dims_a.min() > 0
    xy, valid = series.model.project_volume(torch.tensor([[100.0, 200.0, 300.0]]))
    assert xy.shape == (41, 1, 2) and valid.shape == (41, 1)
    # Global-only differs from full (populated movement grids).
    xy_g, _ = series.model.project_volume_global(torch.tensor([[100.0, 200.0, 300.0]]))
    assert (xy - xy_g).abs().max() > 0.1


def test_strict_loader_rejects_old_format(tmp_path):
    old = WARP_TRIAL / "warp_tiltseries" / f"{TRIAL_SERIES}.xml"
    if not old.exists():
        pytest.skip("warp_trial not available")
    with pytest.raises(ValueError, match="old Warp XML format"):
        load_warp_tiltseries(old)


def test_strict_loader_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<TiltSeries></TiltSeries>")
    with pytest.raises(ValueError, match="missing root attribute"):
        load_warp_tiltseries(bad)
    notxml = tmp_path / "no.xml"
    notxml.write_text("hello")
    with pytest.raises(ValueError, match="not valid XML"):
        load_warp_tiltseries(notxml)


# ---------------------------------------------------------------------------
# w2a global fit
# ---------------------------------------------------------------------------


def test_global_fit_exact_without_level_angles(ts1_xml_path):
    series = load_warp_tiltseries(ts1_xml_path)
    series.ts.level_angle_x = 0.0
    series.ts.level_angle_y = 0.0
    # Rebuild the stripped model after mutating level angles.
    model = type(series.model)(series.ts)
    result = fit_aretomo_globals(model, pixel_size_a=1.7)
    # Without level angles the .aln global family represents Warp's global
    # chain exactly - residual is float32 forward-model noise only.
    assert result.rms_px_heldout < 0.02, result.rms_px_heldout


def test_global_fit_with_level_angles(ts1_xml_path):
    series = load_warp_tiltseries(ts1_xml_path)  # LevelAngleX=1.457, Y=-5.92
    result = fit_aretomo_globals(series.model, pixel_size_a=1.7)

    # The closed-form init alone leaves the LevelAngleX geometry unabsorbed;
    # the fit must improve on it and land at the small unrepresentable rest.
    _rot0, tilt0, _shifts0 = closed_form_init(series.model, 1.7)
    # LevelAngleX (1.457 deg here) leaves an unabsorbable in-plane pre-tilt
    # rotation; the empirical remainder for TS_1 is ~5 px held-out RMS. That
    # error is later baked into the local shifts (w2a step 4).
    assert result.rms_px_heldout < 8.0
    # LevelAngleY folds into TILT: fitted TILT stays close to -(angle+levelY).
    assert (result.tilt_deg - tilt0).abs().max() < 1.0


# ---------------------------------------------------------------------------
# Real-data sign calibration (warp_trial)
# ---------------------------------------------------------------------------


@pytest.fixture
def warp_trial_series(tmp_path):
    xml = WARP_TRIAL / "warp_tiltseries" / f"{TRIAL_SERIES}.xml"
    aln = WARP_TRIAL / "portal_data" / TRIAL_SERIES / "Alignments" / "100" / f"{TRIAL_SERIES}.aln"
    if not xml.exists() or not aln.exists():
        pytest.skip("warp_trial not available")

    # Dim-patch a COPY (old-format XML lacks the dimension attributes).
    root = etree.fromstring(xml.read_bytes())
    img_a = [d * TRIAL_PIX for d in TRIAL_IMG_PX]
    vol_a = [d * TRIAL_PIX for d in TRIAL_VOL_PX]
    root.set("ImageDimensionsAngstrom", f"{img_a[0]}, {img_a[1]}")
    root.set("VolumeDimensionsAngstrom", f"{vol_a[0]}, {vol_a[1]}, {vol_a[2]}")
    patched = tmp_path / f"{TRIAL_SERIES}.xml"
    patched.write_bytes(etree.tostring(root, xml_declaration=True, encoding="utf-8"))

    series = load_warp_tiltseries(patched)
    aln_series = load_aln(aln, pixel_size_a=TRIAL_PIX, volume_dims_a=tuple(vol_a))
    return series, aln_series


def test_tilt_matching(warp_trial_series):
    series, aln_series = warp_trial_series
    match = match_tilts(series.ts.angles, series.ts.use_tilt, aln_series)
    assert len(match.aln_to_warp) == 31
    assert match.max_angle_error_deg < 0.1
    # Bijection over all Warp tilts; for this series Warp's -Angle sequence is
    # ascending like the .aln TILT column, so the mapping is the identity.
    assert sorted(match.aln_to_warp) == list(range(31))
    assert match.aln_to_warp == list(range(31))


def test_calibrate_signs_real_data(warp_trial_series):
    series, aln_series = warp_trial_series
    match = match_tilts(series.ts.angles, series.ts.use_tilt, aln_series)
    result = calibrate_signs(series.model, aln_series, match)
    assert result.configured_is_best
    assert result.best == CONFIGURED
    # Wrong-sign candidates must be far worse than the configured one.
    assert result.margin < 0.5, result.rms_by_candidate
    # And the configured RMS itself should be small (independent alignments of
    # the same stack agree globally to tens of Angstrom).
    assert result.rms_by_candidate[CONFIGURED] < 100.0
