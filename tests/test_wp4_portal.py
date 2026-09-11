"""WP4: portal runs as AreTomo3 sources — data-only resolver tests, plus live
API tests under CETS_NONRIGID_PORTAL_NETWORK=1."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import starfile
import test_r4_a2r as helpers
import torch
from click.testing import CliRunner

from cets_nonrigid.cli import main
from cets_nonrigid.meta import resolve_series
from cets_nonrigid.meta.portal import (
    PortalRunData,
    PortalSection,
    PortalTomogram,
    discover_portal_series,
    https_to_s3,
    implied_voxel,
    pick_tomogram,
)

NET = os.environ.get("CETS_NONRIGID_PORTAL_NETWORK") == "1"
live = pytest.mark.skipif(not NET, reason="set CETS_NONRIGID_PORTAL_NETWORK=1 for live portal tests")


def _synthetic_portal(tmp_path):
    aln_path, tilts_raw = helpers._write_synthetic_aln(tmp_path / "TS_P.aln")
    acq = np.argsort(np.argsort(np.abs(tilts_raw), kind="stable"))  # 0-based, dose-symmetric
    sections = [
        PortalSection(
            z_index=i,
            raw_angle=float(tilts_raw[i]),
            acquisition_order=int(acq[i]),
            exposure_dose=3.0,
            accumulated_dose=3.0 * int(acq[i]),
            frame_name=f"frames/TS_P_{i + 1:03d}_{tilts_raw[i]:.2f}.eer",
            major_defocus_a=21000 + 100 * i,
            minor_defocus_a=20500 + 100 * i,
            astigmatic_angle_deg=30.0,
            phase_shift_rad=0.0,
        )
        for i in range(helpers.T_RAW)
    ]
    data = PortalRunData(
        dataset_id=10445,
        run_id=17687,
        run_name="TS_P",
        https_prefix="https://files.cryoetdataportal.cziscience.com/10445/TS_P/",
        tiltseries_id=1,
        pixel_spacing=helpers.PIX,
        size=(helpers.IMG[0], helpers.IMG[1], helpers.T_RAW),
        voltage_kv=300.0,
        cs_mm=2.7,
        tilt_axis_deg=-96.0,
        https_mrc_file=None,
        https_omezarr_dir="https://files.cryoetdataportal.cziscience.com/10445/TS_P/TiltSeries/100/TS_P.zarr",
        https_angle_list=None,
        alignment_id=5,
        alignment_type="LOCAL",
        https_alignment_file=None,
        alignment_volume_a=(helpers.IMG[0] * helpers.PIX, helpers.IMG[1] * helpers.PIX, 22 * helpers.PIX),
        mdoc_url=None,
        sections=sections,
        tomograms=[
            PortalTomogram(id=1, voxel_spacing=4.99, size=(48, 48, 12), processing="filtered"),
            PortalTomogram(id=2, voxel_spacing=10.0, size=(24, 24, 6), processing="denoised"),
        ],
    )
    return data, aln_path, tilts_raw


def test_portal_resolver_data_only(tmp_path):
    data, aln_path, tilts_raw = _synthetic_portal(tmp_path)
    d = discover_portal_series(data, aln_path, cache_dir=tmp_path / "cache")
    m = resolve_series(d, {"amplitude_contrast": 0.07, "defocus_hand": -1})
    assert m.pixel_size_a == helpers.PIX and m.get_provenance("pixel_size_a").kind == "portal"
    assert m.image_dims_px == (96, 96) and m.n_raw_sections == helpers.T_RAW
    assert (m.voltage_kv, m.cs_mm) == (300.0, 2.7)
    # tomogram box from the finest tomogram: 12 slices x implied voxel (2*96/48 = 4.0 A, not the rounded 4.99) / pix
    assert d.facts["portal_voxel_a"] == pytest.approx(4.0)
    assert m.tomo_dims_px == (96, 96, 24) and "tomogram 1" in m.get_provenance("tomo_dims_px").source
    # dose: portal exposure trusted, exclusive accumulated sequence reproduced
    assert m.dose_per_tilt == pytest.approx(3.0) and m.raw_dose.convention == "exclusive"
    order = torch.argsort(m.raw_dose.acq_index_1b)
    np.testing.assert_allclose(m.raw_dose.pre_exposure[order].numpy(), 3.0 * np.arange(helpers.T_RAW), atol=1e-9)
    assert m.stage_tilt_deg == pytest.approx(list(tilts_raw))
    assert m.tilt_image_names[0].startswith("TS_P_001")
    assert Path(m.ctf_path).name == "TS_P_CTF.txt" and m.get_provenance("ctf_path").kind == "portal"
    from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile

    ctf = AreTomoCtfFile.from_file(m.ctf_path)
    assert ctf.n_rows == helpers.T_RAW and ctf.rows[0].df_max_a == 21000
    assert ctf.rows[0].df_hand in (None, 1)  # unknown hand is written as +1 (informational only)
    assert m.aux["tilt_series_uri_s3"] == "s3://cryoet-data-portal-public/10445/TS_P/TiltSeries/100/TS_P.zarr"
    assert https_to_s3("https://elsewhere/x") == "https://elsewhere/x"
    assert pick_tomogram(data, 10.0).id == 2 and implied_voxel(data, pick_tomogram(data, 10.0)) == pytest.approx(8.0)
    with pytest.raises(ValueError, match="voxel spacing"):
        pick_tomogram(data, 7.0)


def test_portal_resolver_box_from_alignment_when_no_tomogram(tmp_path):
    data, aln_path, _ = _synthetic_portal(tmp_path)
    data.tomograms = []
    d = discover_portal_series(data, aln_path, cache_dir=tmp_path / "cache")
    m = resolve_series(d, {})
    assert m.tomo_dims_px == (96, 96, 22)


@live
def test_live_run_metadata_and_a2r_project(tmp_path):
    """Dataset 10445 / TS_105_5: metadata through the API, .aln downloaded,
    ground-truth ribosome picks, py2rely-style project with the OME-Zarr URI."""
    root = tmp_path / "relion"
    from cets_nonrigid import api

    config = tmp_path / "source.json"
    config.write_text(
        json.dumps(
            {"source": {"defocus_handedness": -1, "amplitude_contrast": 0.07, "particles": "portal:cytosolic ribosome"}}
        )
    )
    cets = tmp_path / "source.cets.json"
    res = CliRunner().invoke(
        main, ["to-cets", "aretomo3", "portal:10445/TS_105_5", "-o", str(cets), "--config", str(config)]
    )
    assert res.exit_code == 0, res.output
    bundle = api.read_bundle(cets)
    context = bundle.context
    assert context.image_frames[0].isotropic_spacing == 1.54
    assert context.reference_frame.size_px == (4096, 4096, 1196)
    assert context.rows[0].exposure_dose == pytest.approx(3.87075, rel=1e-5)
    assert all(row.ctf_metadata is not None for row in context.rows)
    parameters = {p.name: json.loads(p.value_json) for p in context.owner.provenance.parameters}
    assert parameters["metadata_source:pixel_size_a"]["kind"] == "portal"
    assert "18956" in parameters["metadata_source:tomo_dims_px"]["source"]
    # Picks retain their identity and implied physical voxel spacing through CETS.
    assert len(bundle.samples.training.point_ids) + len(bundle.samples.heldout.point_ids) > 10
    assert list((tmp_path / ".cets-native-cache").rglob("24mar08a_Position_105_5.aln"))
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"target": {"placeholder_stack": True}}))
    res = CliRunner().invoke(main, ["from-cets", "relion", str(cets), "-o", str(root), "--config", str(target)])
    assert res.exit_code == 0, res.output
    ts = starfile.read(root / "tilt_series" / "TS_105_5.star")
    assert len(ts) == 31 and set(ts["tomoTiltSeriesURI"]) == {
        "s3://cryoet-data-portal-public/10445/TS_105_5/TiltSeries/100/TS_105_5.zarr"
    }
    assert next(iter(ts["rlnMicrographName"])) == "1@tilt_series/TS_105_5_placeholder.mrcs"
    assert float(ts["rlnMicrographPreExposure"].min()) == 0.0
    parts = starfile.read(root / "particles.star")["particles"]
    assert len(parts) > 10 and parts["rlnTomoName"].unique().tolist() == ["TS_105_5"]
    metrics = json.loads(res.output[res.output.index("{") :])
    assert metrics["max_residual_px"] < 1e-3
