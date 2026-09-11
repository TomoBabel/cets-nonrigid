"""CETS-driven global fitting, conditioning and source independence."""

import numpy as np
import pytest
import torch
import test_r4_a2r as a2r_helpers
import test_r4_w2r as w2r_helpers
from cets_nonrigid import api, convert_store as cs
from cets_nonrigid.runtime import runtime_ir
from cets_nonrigid.fit.ir_global import fit_warp_globals_from_ir, fit_aretomo_globals_from_ir

RNG = np.random.default_rng(123)


def _warp_store(tmp_path, name="src", **xml_kwargs):
    xml = w2r_helpers._write_synthetic_xml(tmp_path / (name + ".xml"), **xml_kwargs)
    bundle = api.to_cets("warp", xml, pixel_size_a=w2r_helpers.PIX)
    path = tmp_path / (name + ".cets.json")
    api.write_bundle(bundle, path)
    return xml, path, runtime_ir(bundle)


def test_global_fitters_golden_and_gauge(tmp_path):
    _, path, ir = _warp_store(tmp_path)
    fit, diag = fit_warp_globals_from_ir(ir)
    assert diag.global_validation_status == "evaluated"
    assert diag.rms_px_train < 1e-3 and diag.rms_px_heldout < 1e-3
    assert fit.level_angle_x_deg == pytest.approx(2.0, abs=1e-3)
    assert float(fit.axis_angle_deg.min()) > -180 and float(fit.axis_angle_deg.max()) <= 180
    _, diag = fit_aretomo_globals_from_ir(ir)
    assert diag.rms_px_train < 1.0


def test_agnosticism_identical_cets_different_snapshots(tmp_path):
    _, path, _ = _warp_store(tmp_path)
    bundle = api.read_bundle(path)
    first = api.fit(bundle, "aretomo3")
    bundle.snapshots = {bundle.context.key: {"unrelated": b"not native geometry"}}
    second = api.fit(bundle, "aretomo3")
    assert first.files == second.files


def test_mode_validation_matrix(tmp_path):
    _, path, _ = _warp_store(tmp_path)
    bundle = api.read_bundle(path)
    for options, message in [
        ({"global_mode": "fit", "aln": tmp_path / "a.aln"}, "no source-global context"),
        ({"global_mode": "template"}, "requires -x"),
        ({"global_mode": "aretomo"}, "exactly --aln"),
        ({"global_mode": "relion"}, "requires --optimisation-set"),
    ]:
        with pytest.raises(ValueError, match=message):
            api.fit(bundle, "warp", **options)
    with pytest.raises(ValueError, match="kinds differ"):
        api.fit(bundle, "mcaln")


def test_global_fitter_anisotropic_translated_volume():
    """Initializer normalization: strongly anisotropic dims, non-unit pixel,
    and particles clustered far from the origin."""
    from cets_nonrigid.ir.build import build_ir_tilt_series_from_points
    from cets_nonrigid.ir.core import IRMeta
    from cets_nonrigid.models.aretomo_ts import AretomoTsModel

    pix = 3.7
    vol_px = (400, 60, 30)
    img_px = (400, 400)
    t = 5
    tilt = torch.linspace(-40, 40, t, dtype=torch.float64)
    rot = torch.full((t,), 12.0, dtype=torch.float64)
    shifts = torch.tensor(RNG.uniform(-3, 3, (t, 2)))
    model = AretomoTsModel(
        rot_deg=rot,
        tilt_deg=tilt,
        shifts_px=shifts,
        raw_size_px=img_px,
        pixel_size_a=pix,
        volume_dims_a=tuple(d * pix for d in vol_px),
        local=None,
    )
    meta = IRMeta(
        kind="tilt_series",
        series_name="A",
        pixel_size_image_a=pix,
        image_dims_px=img_px,
        volume_dims_px=vol_px,
        pixel_size_volume_a=pix,
        projection_index=list(range(t)),
        projection_valid=[True] * t,
        projection_order=list(range(t)),
        projection_dose=[0.0] * t,
        projection_angle_deg=[float(a) for a in tilt],
        projection_sec=list(range(1, t + 1)),
        projection_dark=[False] * t,
        source_tool="aretomo3",
        sampling="particles",
        projection_angle_kind=["effective"] * t,
    )
    # particles clustered in a far corner octant (large coordinate offset)
    vol_a = torch.tensor([d * pix for d in vol_px], dtype=torch.float64)
    pos = (torch.tensor(RNG.uniform(0.7, 0.95, (120, 3)))) * vol_a
    ir = build_ir_tilt_series_from_points(
        model, pos, torch.tensor([d * pix for d in img_px], dtype=torch.float64), meta=meta
    )
    fit, diag = fit_aretomo_globals_from_ir(ir)
    assert diag.rms_px_train < 1e-3
    assert (fit.tilt_deg - tilt).abs().max() < 1e-3
    assert (fit.shifts_px - shifts).abs().max() < 1e-3


def test_global_fitter_degenerate_points_fail(tmp_path):
    """Coplanar particles cannot constrain the affine initializer: rank gate."""
    from cets_nonrigid.ir.build import build_ir_tilt_series_from_points

    aln, _ = a2r_helpers._write_synthetic_aln(tmp_path / "s.aln")
    ir0, _ = cs.dump_ir_aretomo(aln, pixel_size_a=a2r_helpers.PIX, tomo_size_px=a2r_helpers.TOMO)
    vol_a = torch.tensor([d * a2r_helpers.PIX for d in a2r_helpers.TOMO], dtype=torch.float64)
    pos = torch.tensor(RNG.uniform(0.2, 0.8, (50, 3))) * vol_a
    pos[:, 2] = vol_a[2] / 2  # exactly coplanar in z
    meta = ir0.meta.model_copy(update={"sampling": "particles"})
    # rebuild a particle IR through the same source model

    from cets_nonrigid.convert import _PermutedAlnModel  # noqa: F401  (model via dump)
    from cets_nonrigid.io.aln import load_aln

    aln_series = load_aln(aln, pixel_size_a=a2r_helpers.PIX, volume_dims_a=tuple(float(v) for v in vol_a))
    model = aln_series.model
    t = model.n_projections
    meta = meta.model_copy(
        update={
            "projection_index": list(range(t)),
            "projection_valid": [True] * t,
            "projection_order": list(range(t)),
            "projection_dose": [0.0] * t,
            "projection_angle_deg": [float(a) for a in aln_series.model.tilt_deg],
            "projection_sec": list(range(1, t + 1)),
            "projection_dark": [False] * t,
            "projection_label": None,
            "projection_angle_kind": ["effective"] * t,
        }
    )
    ir = build_ir_tilt_series_from_points(
        model,
        pos,
        torch.tensor([d * a2r_helpers.PIX for d in a2r_helpers.IMG], dtype=torch.float64),
        meta=meta,
    )
    with pytest.raises(ValueError, match="rank|degenerate"):
        fit_aretomo_globals_from_ir(ir)
