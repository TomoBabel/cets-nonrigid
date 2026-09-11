"""Step 0 of the 3D-displacement plan: model intermediates + golden pins.

* ``WarpTiltSeriesModel.displace_volume`` vs the C# ``GridVolumeWarp*``
  evaluations dumped by ``tools/warpgolden`` (``tiltseries_volwarp.json``,
  ``volume_warp`` block; TiltSeries.cs:410-426).
* ``WarpTiltSeriesModel.ctf_depth`` vs the golden's defocus column
  (``result.Z = defocus + 1e-4 * Z``, TiltSeries.cs:500) — pins the beam-axis
  component of the displacement independently.
* chain consistency ``premovement - M == full`` and ``premovement == rigid(p + d_t)``.
* RELION ``ctf_depth`` (tomogram.cpp:267-285) vs the literal formula, incl.
  hand = -1 and a non-unit defocus slope; particle-set forwarding.
* equal-dose policy: refused at synthesis, at load and at write (serialized <Dose> re-parsed).
* rlnTomoDefocusSlope IO.
"""

from __future__ import annotations

import json
import re
import warnings

import pytest
import torch
from lxml import etree

from cets_nonrigid.io.warp_xml import load_warp_tiltseries, write_alignment_into_template
from cets_nonrigid.models.base import CtfDepthModel, VolumeDeformingModel
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

GOLDEN_XML = "tests/golden/TS_1_volwarp.xml"
GOLDEN_JSON = "tests/golden/tiltseries_volwarp.json"
DEGENERATE_XML = "tests/data/TS_1.xml"


def _golden():
    with open(GOLDEN_JSON) as fh:
        g = json.load(fh)
    n, nt = len(g["points"]), g["n_tilts"]
    pts = torch.tensor(g["points"], dtype=torch.float64)
    pos = torch.tensor(g["positions"], dtype=torch.float64).reshape(n, nt, 3).permute(1, 0, 2)
    vw = torch.tensor(g["volume_warp"], dtype=torch.float64).reshape(n, nt, 3).permute(1, 0, 2)
    return g, pts, pos, vw


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MagnificationCorrection notice on the fixture
        return load_warp_tiltseries(path)


# ---------------------------------------------------------------------------
# Warp: displacement and CTF depth vs the C# golden
# ---------------------------------------------------------------------------


def test_warp_model_implements_protocols():
    m = _load(GOLDEN_XML).model
    assert isinstance(m, VolumeDeformingModel)
    assert isinstance(m, CtfDepthModel)


def test_displace_volume_matches_csharp_volume_warp():
    g, pts, _pos, vw = _golden()
    m = _load(GOLDEN_XML).model
    assert list(m.ts.grid_volume_warp_x.dimensions) == g["volume_warp_dims"][0] == [3, 3, 2, 4]
    d = m.displace_volume(pts)
    assert d.shape == (g["n_tilts"], pts.shape[0], 3) and d.dtype == torch.float32
    err = (d.to(torch.float64) - vw).abs().max().item()
    assert err <= 1e-5, f"max |d - C#| = {err:.3e} A"
    assert vw.abs().max() > 5.0  # non-degenerate golden: the pin is not vacuous
    # dose coordinates exactly as Warp built them
    dc = m._dose_coords().to(torch.float64)
    assert (dc - torch.tensor(g["dose_coord"], dtype=torch.float64)).abs().max() <= 1e-7


def test_ctf_depth_matches_golden_defocus_column():
    g, pts, pos, _vw = _golden()
    ts = _load(GOLDEN_XML).ts
    m = WarpTiltSeriesModel(ts)
    nt = g["n_tilts"]
    tidx = torch.arange(nt, dtype=torch.float32) / (nt - 1)
    gc = torch.stack([torch.full((nt,), 0.5), torch.full((nt,), 0.5), tidx], -1)
    defocus_t = ts.grid_ctf_defocus.get_interpolated(gc).to(torch.float64)  # micrometres
    z_golden = (pos[..., 2] - defocus_t[:, None]) * 1e4  # Angstrom
    z_model = m.ctf_depth(pts).to(torch.float64)
    err = (z_model - z_golden).abs().max().item()
    # float32 micrometre quantization of a ~3 um defocus is ~2.4e-3 A
    assert err <= 1e-2, f"max |Z - golden| = {err:.3e} A over |Z| <= {z_golden.abs().max():.0f} A"
    # explicit displacement == own displacement; a different field changes the depth
    assert torch.equal(m.ctf_depth(pts, displacement=m.displace_volume(pts)), z_model.to(torch.float32))
    zero = torch.zeros(nt, pts.shape[0], 3)
    assert (m.ctf_depth(pts, displacement=zero).to(torch.float64) - z_model).abs().max() > 1.0


def test_chain_consistency_premovement_and_rigid():
    _g, pts, pos, _vw = _golden()
    ts = _load(GOLDEN_XML).ts
    m = WarpTiltSeriesModel(ts)
    nt = ts.n_tilts
    xy_full, _ = m.project_volume(pts)
    xy_pre, _ = m.project_volume_premovement(pts)
    assert (xy_full.to(torch.float64) - pos[..., :2]).abs().max() <= 1e-2
    d = m.displace_volume(pts)
    # premovement == rigid projection of the per-tilt warped point
    rigid = torch.stack([m.project_volume_global(pts + d[t].to(torch.float64))[0][t] for t in range(nt)])
    assert (xy_pre.to(torch.float64) - rigid.to(torch.float64)).abs().max() <= 1e-2
    # premovement - movement(premovement) == full
    img = ts.image_dimensions_physical.to(torch.float32)
    tidx = torch.arange(nt, dtype=torch.float32) / (nt - 1)
    coords = torch.stack(
        [xy_pre[..., 0] / img[0], xy_pre[..., 1] / img[1], tidx[:, None].expand(-1, pts.shape[0])], -1
    ).reshape(-1, 3)
    mx = ts.grid_movement_x.get_interpolated(coords).reshape(nt, -1)
    my = ts.grid_movement_y.get_interpolated(coords).reshape(nt, -1)
    recon = torch.stack([xy_pre[..., 0] - mx, xy_pre[..., 1] - my], -1)
    assert (recon.to(torch.float64) - xy_full.to(torch.float64)).abs().max() <= 1e-2


def test_degenerate_grids_give_zero_displacement():
    m = _load(DEGENERATE_XML).model
    pts = torch.rand(20, 3, dtype=torch.float64) * m.volume_dims_a.to(torch.float64)
    d = m.displace_volume(pts)
    assert torch.equal(d, torch.zeros_like(d))


def test_ctf_depth_inverted_angles_matches_warpylib_defocus_channel():
    """Consistency (not a C# golden): the inverted branch of ctf_depth against
    warpylib's own defocus channel (positions.py:299-313, a transcription of
    TiltSeries.cs:472-480)."""
    from warpylib.tilt_series.positions import get_position_in_all_tilts_single

    _g, pts, _pos, _vw = _golden()
    ts = _load(GOLDEN_XML).ts
    ts.are_angles_inverted = True
    m = WarpTiltSeriesModel(ts)
    nt = ts.n_tilts
    out = get_position_in_all_tilts_single(ts, pts.to(torch.float32))  # (N, T, 3)
    tidx = torch.arange(nt, dtype=torch.float32) / (nt - 1)
    gc = torch.stack([torch.full((nt,), 0.5), torch.full((nt,), 0.5), tidx], -1)
    defocus_t = ts.grid_ctf_defocus.get_interpolated(gc).to(torch.float64)
    z_ref = (out[..., 2].permute(1, 0).to(torch.float64) - defocus_t[:, None]) * 1e4
    z_model = m.ctf_depth(pts).to(torch.float64)
    assert (z_model - z_ref).abs().max() <= 1e-2
    # and it differs from the non-inverted depth (the flip is exercised)
    ts.are_angles_inverted = False
    assert (WarpTiltSeriesModel(ts).ctf_depth(pts).to(torch.float64) - z_model).abs().max() > 10.0


# ---------------------------------------------------------------------------
# Equal-dose policy
# ---------------------------------------------------------------------------


def _with_equal_dose(xml_bytes: bytes) -> bytes:
    root = etree.fromstring(xml_bytes)
    dose = root.find("Dose")
    n = len(dose.text.split())
    dose.text = "\n".join(["12.5"] * n)
    return etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)


def test_equal_dose_is_refused_at_load_with_a_nudge(tmp_path):
    """An equal-dose series is invalid Warp metadata whatever the grids (verified
    with WarpTools 2.0.0: ts_reconstruct dies in GetCTFsForOneParticle); the
    strict loader refuses it and says where a dose comes from."""
    for name, src in (("volwarp", GOLDEN_XML), ("degenerate", DEGENERATE_XML)):
        bad = tmp_path / f"{name}_equal_dose.xml"
        bad.write_bytes(_with_equal_dose(_load(src).xml_bytes))
        with pytest.raises(ValueError, match="same cumulative dose.*--dose-per-tilt"):
            _load(bad)


def test_synthesis_without_dose_is_refused():
    from cets_nonrigid.io.warp_synth import synthesize_tilt_series

    t = 4
    with pytest.raises(ValueError, match="without a per-tilt dose.*--dose-per-tilt"):
        synthesize_tilt_series(
            angles_deg=[-30, -10, 10, 30], use_tilt=[1, 1, 1, 1],
            axis_angles_deg=[85.0] * t, axis_offset_x_a=[0.0] * t, axis_offset_y_a=[0.0] * t,
            image_dims_a=(192.0, 192.0), volume_dims_a=(96.0, 96.0, 48.0), pixel_size_a=2.0,
        )


def test_writer_validates_serialized_dose_and_leaves_no_file(tmp_path):
    """The template's <Dose> is what the output carries; the check re-parses the
    serialized bytes (in a temp file, before publication) and a rejection
    publishes nothing — whatever the grids."""
    from warpylib import LinearGrid4D

    src = _load(DEGENERATE_XML)
    template = _with_equal_dose(src.xml_bytes)  # equal doses in the TEMPLATE
    ts = src.ts  # its own dose is NOT equal; only the serialized template dose counts
    assert float(ts.max_dose - ts.min_dose) > 0
    out = tmp_path / "out.xml"
    with pytest.raises(ValueError, match="refusing to write.*same cumulative dose.*--dose-per-tilt"):
        write_alignment_into_template(template, ts, out)
    assert not out.exists() and not list(tmp_path.glob("*.tmp-*"))
    ts.grid_volume_warp_x = LinearGrid4D((2, 2, 1, 3), torch.arange(12, dtype=torch.float32))
    with pytest.raises(ValueError, match="refusing to write"):
        write_alignment_into_template(template, ts, out)
    assert not out.exists()
    # a template with real doses writes fine
    write_alignment_into_template(src.xml_bytes, ts, out)
    assert out.exists() and len(set(re.findall(r"[\d.]+", etree.parse(str(out)).getroot().find("Dose").text))) > 1


# ---------------------------------------------------------------------------
# RELION: ctf_depth convention, forwarding, defocus-slope IO
# ---------------------------------------------------------------------------


def _relion_model(hand=1, slope=1.0):
    from cets_nonrigid.models.relion_ts import RelionTomogramModel

    t = 5
    return RelionTomogramModel(
        xtilt_deg=torch.full((t,), 1.5), ytilt_deg=torch.linspace(-60, 60, t),
        zrot_deg=torch.full((t,), 85.0), xshift_a=torch.linspace(-3, 3, t), yshift_a=torch.zeros(t),
        tomo_dims_px=(97, 96, 47), image_dims_px=(120, 110), pixel_size_a=1.7,
        hand=hand, defocus_slope=slope,
    )


@pytest.mark.parametrize("hand,slope", [(1, 1.0), (-1, 1.0), (1, 1.3), (-1, 0.8)])
def test_relion_ctf_depth_literal_formula(hand, slope):
    """tomogram.cpp:267-280: dz = hand * pixelSize * defocusSlope *
    ((P_f pos).z - (P_f centre).z), pos in decentered px, FLOAT centre."""
    m = _relion_model(hand, slope)
    assert isinstance(m, CtfDepthModel) and not isinstance(m, VolumeDeformingModel)
    pts = torch.rand(7, 3, dtype=torch.float64) * m.volume_dims_a
    got = m.ctf_depth(pts)
    p = m.projection_matrices
    pos_h = torch.cat([pts / m.pixel_size_a, torch.ones(7, 1, dtype=torch.float64)], -1)  # (N, 4)
    cen_h = torch.tensor([97 / 2, 96 / 2, 47 / 2, 1.0], dtype=torch.float64)
    depth_px = torch.einsum("tij,nj->tni", p, pos_h)[..., 2] - torch.einsum("tij,j->ti", p, cen_h)[:, None, 2]
    ref = hand * m.pixel_size_a * slope * depth_px
    assert (got - ref).abs().max() <= 1e-9
    assert torch.equal(m.ctf_depth(pts, displacement=torch.ones(5, 7, 3)), got)  # trajectories never enter
    # matrix-based construction carries the same convention
    from cets_nonrigid.models.relion_ts import RelionTomogramModel

    mm = RelionTomogramModel.from_matrices(
        p, pixel_size_a=m.pixel_size_a, tomo_dims_px=m.tomo_dims_px, image_dims_px=m.image_dims_px,
        hand=hand, defocus_slope=slope,
    )
    assert (mm.ctf_depth(pts) - ref).abs().max() <= 1e-9
    assert (m.with_ctf_convention(hand=-hand).ctf_depth(pts) + ref).abs().max() <= 1e-9


def test_relion_particle_set_forwards_displacement_and_ctf_depth():
    from cets_nonrigid.models.relion_ts import RelionParticleSetModel

    g = _relion_model(-1, 1.1)
    pos = torch.rand(6, 3, dtype=torch.float64) * g.volume_dims_a
    traj = torch.randn(5, 6, 3, dtype=torch.float64)
    with_traj = RelionParticleSetModel(g, pos, trajectories_a=traj)
    without = RelionParticleSetModel(g, pos)
    assert torch.equal(with_traj.displace_volume(pos), traj)
    assert without.displace_volume(pos) is None
    # CTF depth availability is independent of the trajectories and uses the STATIC positions
    for m in (with_traj, without):
        assert torch.equal(m.ctf_depth(pos), g.ctf_depth(pos))
    # full projection == rigid projection of pos + trajectory (2D), to float64 precision
    xy, _ = with_traj.project_volume(pos)
    r = g.projection_matrices[:, :3, :3]
    t = g.projection_matrices[:, :3, 3]
    ref = (torch.einsum("tij,tpj->tpi", r, (pos[None] + traj) / g.pixel_size_a) + t[:, None, :])[..., :2]
    assert (xy - ref * g.pixel_size_a).abs().max() <= 1e-9


def test_defocus_slope_star_io(tmp_path):
    import starfile

    from cets_nonrigid.io.relion_star import read_tomograms_star

    bundle = _w2r_bundle(tmp_path / "b0")
    data = read_tomograms_star(bundle.tomograms_star)
    (tomo,) = data.values()
    assert tomo.defocus_slope is None  # not emitted from a Warp source -> RELION default 1.0
    glob = starfile.read(bundle.tomograms_star, always_dict=True)["global"]
    assert "rlnTomoDefocusSlope" not in glob.columns
    # inject the column, read it back, and re-emit it
    glob["rlnTomoDefocusSlope"] = 1.25
    edited = tmp_path / "edited.star"
    starfile.write({"global": glob}, edited)
    (tomo2,) = read_tomograms_star(edited).values()
    assert tomo2.defocus_slope == 1.25
    from cets_nonrigid.convert_relion import relion_model_from_data

    model = relion_model_from_data(tomo2, image_dims_px=(96, 96))
    assert model.defocus_slope == 1.25 and model.hand == tomo2.hand
    from cets_nonrigid.io.relion_star import tomogram_global_row

    row = tomogram_global_row(tomo2, "x.star")
    assert row["rlnTomoDefocusSlope"] == 1.25
    assert "rlnTomoDefocusSlope" not in tomogram_global_row(tomo, "x.star")


def _w2r_bundle(root):
    import test_r4_w2r as h

    from cets_nonrigid.convert_relion import warp_to_relion

    root.mkdir()
    xml = h._write_synthetic_xml(root / "src.xml")
    stack = h._dummy_stack(root / "stack.mrc")
    pos = h._particles()
    names = [f"TS/{i + 1}" for i in range(pos.shape[0])]
    return warp_to_relion(
        xml, root / "bundle", pixel_size_a=h.PIX, tomo_name="TS",
        positions_eff_a=pos, particle_names=names, tilt_stack=stack,
    )
