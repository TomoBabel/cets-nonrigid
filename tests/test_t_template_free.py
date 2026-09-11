"""Workstream T: template-free Warp conversion (a2w, r2w, m2w, rm2w).

Contracts under test (plan rev. 5): synthesized templates behave exactly like
loaded ones downstream; geometry is never guessed; generated CTF carries every
physically known field (never warpylib's 1.0-A default pixel size) with 1x1xT
placeholder grids; MoviePath keeps per-row correspondence; serialization is
atomic (a failed validation leaves nothing behind); defaults are loud and
recorded; template mode is unchanged and rejects synthesis-only options.
"""

import numpy as np
import pytest
import test_r4_a2r as a2r_helpers
import test_r4_w2r as w2r_helpers
import torch

from cets_nonrigid.convert import aretomo_to_warp, mcaln_to_warp_movie, warp_movie_to_mcaln
from cets_nonrigid.io.warp_movie_xml import load_warp_movie_strict
from cets_nonrigid.io.warp_synth import atomic_write_validated
from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

RNG = np.random.default_rng(20260910)

PIX = a2r_helpers.PIX
T_RAW = a2r_helpers.T_RAW
DARK_SEC0 = a2r_helpers.DARK_SEC0
TOMO = a2r_helpers.TOMO
IMG = a2r_helpers.IMG


def _synthetic_aln(tmp_path):
    path, tilts_raw = a2r_helpers._write_synthetic_aln(tmp_path / "src.aln")
    return path, tilts_raw


def _dose(tilts_raw):
    return a2r_helpers.synthetic_raw_dose(tilts_raw)


# ---------------------------------------------------------------------------
# a2w template-free
# ---------------------------------------------------------------------------


def test_a2w_template_free_end_to_end(tmp_path):
    aln_path, tilts_raw = _synthetic_aln(tmp_path)
    out = tmp_path / "out.xml"

    # no dose source: refused (an equal-dose XML is invalid Warp metadata), with a nudge
    with pytest.raises(ValueError, match="without a per-tilt dose.*--dose-per-tilt"):
        aretomo_to_warp(aln_path, None, out, pixel_size_a=PIX, tomo_size_px=TOMO)

    with pytest.warns(UserWarning, match="generated"):
        r = aretomo_to_warp(
            aln_path, None, out, pixel_size_a=PIX, tomo_size_px=TOMO, raw_dose=_dose(tilts_raw),
        )

    assert r.template_source == "generated"
    assert "zero dose" not in r.defaulted_fields
    assert "blank movie paths" in r.defaulted_fields
    assert "AreAnglesInverted=False" in r.defaulted_fields

    series = load_warp_tiltseries(out)  # strict reload
    ts = series.ts
    # dark-row reconstruction: raw-section order incl. the dark row
    assert ts.n_tilts == T_RAW
    assert not bool(ts.use_tilt[DARK_SEC0])
    assert int(ts.use_tilt.sum()) == T_RAW - 1
    # dimensions from RawSize x pix and --tomo-size x pix
    assert torch.allclose(
        ts.image_dimensions_physical, torch.tensor([IMG[0] * PIX, IMG[1] * PIX])
    )
    assert torch.allclose(
        ts.volume_dimensions_physical,
        torch.tensor([TOMO[0] * PIX, TOMO[1] * PIX, TOMO[2] * PIX]),
    )
    # generated placeholder CTF: physically known pixel size, 1x1xT grids
    assert float(ts.ctf.pixel_size) == pytest.approx(PIX)
    assert tuple(ts.grid_ctf_defocus.dimensions) == (1, 1, T_RAW)
    # projection fidelity vs the source .aln model on regular rows
    from cets_nonrigid.io.aln import load_aln
    from cets_nonrigid.ir.sampling import volume_grid

    aln_series = load_aln(
        aln_path, pixel_size_a=PIX, volume_dims_a=tuple(float(v) * PIX for v in TOMO)
    )
    out_model = WarpTiltSeriesModel(ts)
    pts = volume_grid(out_model.volume_dims_a.to(torch.float64), (5, 5, 3))
    w_xy, w_valid = out_model.project_volume(pts)
    a_xy, a_valid = aln_series.model.project_volume(pts)
    perm = torch.tensor([int(m) for m in r.match.aln_to_warp])
    both = w_valid[perm] & a_valid
    err = (w_xy[perm] - a_xy)[both].norm(dim=-1)
    assert float(err.median()) / PIX < 2.5  # representation cost, not misalignment

    # Native disposition remains checked here; CETS snapshots and persisted fit
    # reports are covered in test_cets_exchange and test_displacement_cli.
    assert r.template_source == "generated"


def test_a2w_template_free_dose_paths_and_ctf(tmp_path):
    aln_path, _tilts_raw = _synthetic_aln(tmp_path)
    tlt, ctf_txt = a2r_helpers._synthetic_inputs(tmp_path)[1:3]

    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

    from cets_nonrigid.io.aln import raw_tilts_from_aln
    from cets_nonrigid.io.dose import raw_dose_from_tlt

    raw_dose = raw_dose_from_tlt(tlt)
    tilt_images = [f"frames/img_{i:02d}.mrc" for i in range(T_RAW)]

    out = tmp_path / "out.xml"
    with pytest.warns(UserWarning, match="generated"):
        r = aretomo_to_warp(
            aln_path, None, out, pixel_size_a=PIX, tomo_size_px=TOMO,
            raw_dose=raw_dose, tilt_images=tilt_images,
            ctf_file=ctf_txt, ctf_voltage_kv=300.0, ctf_cs_mm=2.7, ctf_amp_contrast=0.07,
        )
    assert "zero dose" not in r.defaulted_fields
    assert "blank movie paths" not in r.defaulted_fields
    assert "placeholder per-tilt CTF" not in r.defaulted_fields
    assert not any("CTF scalars" in f for f in r.defaulted_fields)

    ts = load_warp_tiltseries(out).ts
    assert torch.allclose(ts.dose, raw_dose.pre_exposure.to(torch.float32))
    assert ts.tilt_movie_paths == tilt_images  # exactly one row per raw section
    assert float(ts.ctf.voltage) == 300.0
    # real per-tilt CTF grids, not placeholders
    assert float(ts.grid_ctf_defocus.values.flatten()[0]) > 0.5  # um, genuine defocus

    # verify raw_tilts consistency guard still holds
    assert raw_tilts_from_aln(AreTomo3ALN.from_file(str(aln_path))).shape[0] == T_RAW


def test_a2w_template_vs_template_free_equivalence(tmp_path):
    """Same closed-form target + same IR grid: the fitted movement grids must
    agree between template mode (a synthesized-but-saved template) and
    template-free mode."""
    aln_path, tilts_raw = _synthetic_aln(tmp_path)

    with pytest.warns(UserWarning, match="generated"):
        r_free = aretomo_to_warp(
            aln_path, None, tmp_path / "free.xml", pixel_size_a=PIX, tomo_size_px=TOMO, raw_dose=_dose(tilts_raw)
        )
    # write the SAME synthesized template to disk and run template mode with it
    # (an a2w OUTPUT is not a nominal-stage template: its angles bake in the
    # .aln AlphaOffset per the pinned Warp convention)
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

    from cets_nonrigid.convert import _synthesize_a2w_template

    template_series, _rep = _synthesize_a2w_template(
        AreTomo3ALN.from_file(str(aln_path)),
        pixel_size_a=PIX, tomo_size_px=TOMO, raw_dose=_dose(tilts_raw), tilt_images=None,
        angles_inverted=None, voltage_kv=None, cs_mm=None, amplitude_contrast=None,
    )
    tmpl_xml = tmp_path / "template.xml"
    tmpl_xml.write_bytes(template_series.xml_bytes)
    r_tmpl = aretomo_to_warp(aln_path, tmpl_xml, tmp_path / "tmpl.xml")

    ts_a = load_warp_tiltseries(tmp_path / "free.xml").ts
    ts_b = load_warp_tiltseries(tmp_path / "tmpl.xml").ts
    torch.testing.assert_close(
        ts_a.grid_movement_x.values, ts_b.grid_movement_x.values, atol=1e-4, rtol=0
    )
    torch.testing.assert_close(
        ts_a.grid_movement_y.values, ts_b.grid_movement_y.values, atol=1e-4, rtol=0
    )
    assert r_free.fit.rms_a_heldout == pytest.approx(r_tmpl.fit.rms_a_heldout, abs=1e-3)


def test_a2w_conflicts_and_geometry(tmp_path):
    aln_path, tilts_raw = _synthetic_aln(tmp_path)
    xml = tmp_path / "template.xml"
    with pytest.warns(UserWarning):
        aretomo_to_warp(aln_path, None, xml, pixel_size_a=PIX, tomo_size_px=TOMO, raw_dose=_dose(tilts_raw))

    # template + synthesis-only options -> rejected, never silently ignored
    with pytest.raises(ValueError, match="synthesis-only"):
        aretomo_to_warp(aln_path, xml, tmp_path / "x.xml", pixel_size_a=PIX)
    with pytest.raises(ValueError, match="synthesis-only"):
        aretomo_to_warp(aln_path, xml, tmp_path / "x.xml", angles_inverted=True)
    # missing geometry
    with pytest.raises(ValueError, match="pixel_size_a and tomo_size_px"):
        aretomo_to_warp(aln_path, None, tmp_path / "x.xml", pixel_size_a=PIX)
    # existing output refused (atomic path)
    import warnings as _warnings

    with pytest.raises(FileExistsError), _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        aretomo_to_warp(aln_path, None, xml, pixel_size_a=PIX, tomo_size_px=TOMO, raw_dose=_dose(tilts_raw))
    # partially specified tilt images refused
    with pytest.raises(ValueError, match="partially specified|every raw section"), \
            _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        aretomo_to_warp(
            aln_path, None, tmp_path / "y.xml", pixel_size_a=PIX, tomo_size_px=TOMO, raw_dose=_dose(tilts_raw),
            tilt_images=["a.mrc"] * (T_RAW - 1) + [""],
        )


# ---------------------------------------------------------------------------
# r2w template-free
# ---------------------------------------------------------------------------


def test_r2w_template_free_roundtrip(tmp_path):
    """A Warp-generated RELION bundle through template-free r2w: dims, dose,
    CTF, handedness, and projection fidelity all come from the star."""
    from cets_nonrigid.convert_relion import relion_to_warp, warp_to_relion

    xml = w2r_helpers._write_synthetic_xml(tmp_path / "src.xml", inverted=True)
    stack = w2r_helpers._dummy_stack(tmp_path / "stack.mrc")
    vol_a = torch.tensor(w2r_helpers.VOL_A, dtype=torch.float64)
    pos = torch.tensor(RNG.uniform(0.08, 0.92, (200, 3))) * vol_a
    names = [f"TS_TF/{i + 1}" for i in range(pos.shape[0])]
    r = warp_to_relion(
        xml, tmp_path / "bundle", pixel_size_a=w2r_helpers.PIX, tomo_name="TS_TF",
        positions_eff_a=pos, particle_names=names, tilt_stack=stack,
    )

    img_px = (
        round(w2r_helpers.IMG_A[0] / w2r_helpers.PIX),
        round(w2r_helpers.IMG_A[1] / w2r_helpers.PIX),
    )
    out = tmp_path / "generated.xml"
    with pytest.warns(UserWarning, match="generated"):
        res = relion_to_warp(
            None, out, tomo_name="TS_TF",
            optimisation_set=r.optimisation_set,
            movement_grid=(4, 4),
            image_size_px=img_px,
        )
    assert res.template_source == "generated"
    assert res.fit.heldout_status == "evaluated"
    assert res.fit.rms_a_heldout / w2r_helpers.PIX < 0.1  # same gate as the template path

    ts = load_warp_tiltseries(out).ts
    assert torch.allclose(
        ts.image_dimensions_physical, torch.tensor(w2r_helpers.IMG_A)
    )
    assert torch.allclose(
        ts.volume_dimensions_physical, torch.tensor(w2r_helpers.VOL_A)
    )
    # handedness derived from rlnTomoHand (source was inverted=True)
    assert bool(ts.are_angles_inverted)
    # dose and per-tilt CTF populated from the star
    assert float(ts.dose.max()) > 0
    assert float(ts.ctf.pixel_size) == pytest.approx(w2r_helpers.PIX)
    assert float(ts.ctf.voltage) == 300.0
    assert float(ts.grid_ctf_defocus.values.flatten()[0]) > 1.0  # genuine um defocus
    # dose/paths/CTF must not be in the defaulted list
    assert "zero dose" not in res.defaulted_fields
    assert "placeholder per-tilt CTF" not in res.defaulted_fields


def test_r2w_template_free_requires_image_size(tmp_path):
    from cets_nonrigid.convert_relion import relion_to_warp

    with pytest.raises(ValueError, match="image_size_px"):
        relion_to_warp(None, tmp_path / "o.xml", tomo_name="X", tomograms_star=tmp_path / "t.star")


# ---------------------------------------------------------------------------
# m2w / rm2w template-free
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_movie(tmp_path_factory):
    import test_m6_movie_fits as m6

    golden_xml = m6.GOLDEN_DIR / "movie_synthetic.xml"
    if not golden_xml.exists():
        pytest.skip("synthetic movie fixture missing")
    import json

    with open(m6.GOLDEN_DIR / "movie_positions.json") as f:
        golden = json.load(f)
    tmp = tmp_path_factory.mktemp("m2w")
    mcaln = tmp / "movie.mcaln"
    warp_movie_to_mcaln(
        golden_xml, mcaln,
        image_size_px=m6.FRAME_PX, pixel_size_a=m6.PIX,
        n_frames=golden["n_frames"], fraction_frames=golden["fraction_frames"],
        raw_frames_per_aligned=15,
    )
    return mcaln, golden, m6


def test_m2w_template_free(golden_movie, tmp_path):
    mcaln, golden, m6 = golden_movie
    out = tmp_path / "movie.xml"
    with pytest.warns(UserWarning, match="generated"):
        r = mcaln_to_warp_movie(mcaln, None, out, movie_path="frames/movie.tiff")
    # stem mismatch warned about above ("movie" vs "movie" actually matches);
    # verify the record instead
    assert r.template_source == "generated"

    wx = load_warp_movie_strict(out)  # NEW strict movie loader
    assert float(wx.movie.ctf.pixel_size) == pytest.approx(m6.PIX)

    # trajectory fidelity with externally supplied runtime metadata
    from cets_nonrigid.io.motion_txt import McAln
    from cets_nonrigid.models.warp_movie import WarpMovieModel

    src = McAln.from_file(mcaln)
    # m2w grids are written for FractionFrames = 1 (documented contract)
    model_out = WarpMovieModel.from_xml(
        out,
        n_frames=src.aligned_frame_count,
        image_dims_a=tuple(golden["image_dims"]),
        fraction_frames=1.0,
    )
    model_src = src.to_model()
    dims = torch.tensor(golden["image_dims"], dtype=torch.float64)
    pts = torch.tensor(RNG.uniform(0.1, 0.9, (40, 2))) * dims
    out_xy = model_out.map_image(pts)[0]
    src_xy = model_src.map_image(pts)[0]
    err = (out_xy - src_xy).norm(dim=-1)
    assert float(err.median()) < 3.0  # within the fit's own representation cost


def test_m2w_template_free_stem_warning(golden_movie, tmp_path):
    mcaln, _golden, _m6 = golden_movie
    out = tmp_path / "different_name.xml"
    with pytest.warns(UserWarning, match="does not match"):
        mcaln_to_warp_movie(mcaln, None, out, movie_path="frames/movie.tiff")


def test_rm2w_template_free(golden_movie, tmp_path):
    mcaln, _golden, m6 = golden_movie
    from cets_nonrigid.convert_relion_movie import (
        mcaln_to_relion_motion,
        relion_motion_to_warp_movie,
    )

    star = tmp_path / "motion.star"
    mcaln_to_relion_motion(mcaln, star, voltage_kv=300.0)
    out = tmp_path / "rm2w.xml"
    with pytest.warns(UserWarning, match="generated"):
        r = relion_motion_to_warp_movie(star, None, out)
    assert r.template_source == "generated"
    wx = load_warp_movie_strict(out)
    assert float(wx.movie.ctf.pixel_size) == pytest.approx(m6.PIX)
    assert float(wx.movie.ctf.voltage) == 300.0  # carried from the motion star


# ---------------------------------------------------------------------------
# MoviePath cardinality + atomic serialization
# ---------------------------------------------------------------------------


def test_moviepath_mixed_blank_rows_roundtrip(tmp_path):
    """Blank rows (dark tilts without images) keep per-row correspondence:
    the strict loader parses MoviePath from the raw XML itself (warpylib drops
    blank entries); a count mismatch is rejected strictly."""
    from cets_nonrigid.io.warp_synth import synthesize_tilt_series

    t = 4
    ts, _ = synthesize_tilt_series(
        angles_deg=[-30, -10, 10, 30], use_tilt=[1, 0, 1, 1],
        axis_angles_deg=[85.0] * t, axis_offset_x_a=[0.0] * t, axis_offset_y_a=[0.0] * t,
        image_dims_a=(192.0, 192.0), volume_dims_a=(96.0, 96.0, 48.0), pixel_size_a=2.0,
        dose=[0.0, 3.0, 6.0, 9.0],
    )
    ts.tilt_movie_paths = ["a.mrc", "", "c.mrc", "d.mrc"]  # mixed: blank dark row
    xml = tmp_path / "mixed.xml"
    ts.save_meta(str(xml))
    series = load_warp_tiltseries(xml)
    assert series.ts.tilt_movie_paths == ["a.mrc", "", "c.mrc", "d.mrc"]

    # corrupt the cardinality -> the strict loader must reject it
    from lxml import etree

    root = etree.fromstring(xml.read_bytes())
    root.find("MoviePath").text = "only_one.mrc"
    bad = tmp_path / "bad.xml"
    bad.write_bytes(etree.tostring(root))
    with pytest.raises(ValueError, match="MoviePath"):
        load_warp_tiltseries(bad)


def test_atomic_serialization_cleanup(tmp_path):
    out = tmp_path / "out.xml"

    def bad_writer(tmp):
        from pathlib import Path as _P

        _P(tmp).write_text("<NotATiltSeries/>")

    with pytest.raises(ValueError):
        atomic_write_validated(bad_writer, out, load_warp_tiltseries)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []  # no stray temporaries

    # refuses to overwrite
    out.write_text("existing")
    with pytest.raises(FileExistsError):
        atomic_write_validated(bad_writer, out, load_warp_tiltseries)
    assert out.read_text() == "existing"
