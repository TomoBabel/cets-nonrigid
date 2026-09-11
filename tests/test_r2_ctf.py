"""R2: CTF layer — exact value maps (G10), file IO, and the G4 hand golden."""

import copy

import numpy as np
import pytest
import torch

from cets_nonrigid.ctf import (
    TiltCtf,
    aretomo_raw_order_permutation,
    canonicalize_astigmatism,
    tiltctf_from_aretomo,
    tiltctf_from_relion_columns,
    tiltctf_from_warp,
    tiltctf_to_aretomo,
    tiltctf_to_relion_columns,
    tiltctf_to_warp,
    warp_has_tilt_ctf,
)
from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile
from cets_nonrigid.io.warp_xml import load_warp_tiltseries

RNG = np.random.default_rng(20260901)


def _random_ctf(t=7, with_diag=True):
    u = torch.tensor(RNG.uniform(15000, 40000, t))
    v = u - torch.tensor(RNG.uniform(0, 900, t))
    return TiltCtf(
        defocus_u_a=u,
        defocus_v_a=v,
        angle_deg=torch.tensor(RNG.uniform(0, 180, t)),
        phase_deg=torch.tensor(RNG.uniform(0, 40, t)),
        score=torch.tensor(RNG.uniform(0, 0.3, t)) if with_diag else None,
        res_a=torch.tensor(RNG.uniform(3, 12, t)) if with_diag else None,
        voltage_kv=300.0,
        cs_mm=2.7,
        amplitude_contrast=0.07,
    )


# --- canonicalization ---------------------------------------------------------


def test_canonicalization_is_a_defocus_field_identity():
    u = torch.tensor([20000.0, 18000.0, 25000.0], dtype=torch.float64)
    v = torch.tensor([21000.0, 17000.0, 25500.0], dtype=torch.float64)  # rows 0, 2 violate U >= V
    a = torch.tensor([170.0, 20.0, -30.0], dtype=torch.float64)
    u2, v2, a2 = canonicalize_astigmatism(u, v, a)
    assert (u2 >= v2).all()
    assert ((a2 >= 0) & (a2 < 180)).all()
    before = TiltCtf.__new__(TiltCtf)  # evaluate the field without re-canonicalizing
    for theta in [0.0, 33.0, 90.0, 145.0]:
        th = torch.full((3,), theta)
        f_before = (u + v) / 2 + (u - v) / 2 * torch.cos(2 * torch.deg2rad(th - a))
        f_after = (u2 + v2) / 2 + (u2 - v2) / 2 * torch.cos(2 * torch.deg2rad(th - a2))
        torch.testing.assert_close(f_before, f_after, atol=1e-9, rtol=0)
    del before


# --- G10: value maps ----------------------------------------------------------


def test_relion_columns_roundtrip_exact():
    ctf = _random_ctf()
    cols = tiltctf_to_relion_columns(ctf)
    assert set(cols) == {
        "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnPhaseShift",
        "rlnCtfFigureOfMerit", "rlnCtfMaxResolution",
    }
    back = tiltctf_from_relion_columns(cols, voltage_kv=300.0, cs_mm=2.7, amplitude_contrast=0.07)
    torch.testing.assert_close(back.defocus_u_a, ctf.defocus_u_a)
    torch.testing.assert_close(back.defocus_v_a, ctf.defocus_v_a)
    torch.testing.assert_close(back.angle_deg, ctf.angle_deg)
    torch.testing.assert_close(back.phase_deg, ctf.phase_deg)
    # optional diagnostics stay optional
    ctf2 = _random_ctf(with_diag=False)
    cols2 = tiltctf_to_relion_columns(ctf2)
    assert "rlnCtfFigureOfMerit" not in cols2 and "rlnCtfMaxResolution" not in cols2


def test_aretomo_roundtrip_through_file_format():
    t = 7
    ctf = _random_ctf(t)
    angles = torch.tensor(RNG.permutation(np.linspace(-60, 60, t)))  # XML order != sorted
    f = tiltctf_to_aretomo(ctf, angles, df_hand=1)
    text = f.to_string()
    parsed = AreTomoCtfFile.from_string(text)
    back = tiltctf_from_aretomo(parsed, angles)
    # %8.2f defocus/angle -> 0.005 quantization; %9.4f rad phase -> ~3e-3 deg
    torch.testing.assert_close(back.defocus_u_a, ctf.defocus_u_a, atol=6e-3, rtol=0)
    torch.testing.assert_close(back.defocus_v_a, ctf.defocus_v_a, atol=6e-3, rtol=0)
    torch.testing.assert_close(back.angle_deg, ctf.angle_deg, atol=6e-3, rtol=0)
    torch.testing.assert_close(back.phase_deg, ctf.phase_deg, atol=4e-3, rtol=0)


def test_aretomo_row_order_is_descending_warp_angle():
    # AreTomo sorts ascending by ITS tilt (= -Warp angle): raw order is
    # DESCENDING Warp angle (verified on 24jul16a .aln SEC/TILT vs XML Angles).
    t = 5
    ctf = _random_ctf(t)
    angles = torch.tensor([30.0, -30.0, 0.0, 60.0, -60.0])  # XML order
    f = tiltctf_to_aretomo(ctf, angles)
    perm = aretomo_raw_order_permutation(angles)
    assert perm.tolist() == [3, 0, 2, 1, 4]
    for i in range(t):
        assert f.rows[i].micrograph == i + 1
        assert f.rows[i].df_max_a == pytest.approx(float(ctf.defocus_u_a[perm[i]]), abs=1e-9)


def test_phase_shift_unit_matrix():
    # a 90-degree phase plate: 90 deg == pi/2 rad (AreTomo file) == 0.5 pi (Warp)
    ctf = TiltCtf(
        defocus_u_a=torch.tensor([20000.0]),
        defocus_v_a=torch.tensor([19000.0]),
        angle_deg=torch.tensor([45.0]),
        phase_deg=torch.tensor([90.0]),
    )
    f = tiltctf_to_aretomo(ctf, torch.tensor([0.0]))
    assert f.rows[0].phase_rad == pytest.approx(np.pi / 2, abs=1e-9)
    cols = tiltctf_to_relion_columns(ctf)
    assert cols["rlnPhaseShift"][0] == pytest.approx(90.0)


# --- _CTF.txt parsing/validation ----------------------------------------------


def test_ctf_file_parses_aretomo_format_and_zero_base():
    text = (
        "# Columns: #1 micrograph number; ...\n"
        "   1 21000.50 20000.25    12.00    1.5708   0.1500   4.5000   1\n"
        "   2 22000.00 21500.00   170.00    0.0000   0.2000   5.0000   1\n"
    )
    f = AreTomoCtfFile.from_string(text)
    assert f.n_rows == 2
    assert f.rows[0].df_max_a == 21000.50
    assert f.rows[0].phase_rad == pytest.approx(1.5708)
    # zero-based numbering accepted via min-normalization (CLoadCtfResults.cpp:84-94)
    f0 = AreTomoCtfFile.from_string(text.replace("   1 ", "   0 ").replace("   2 ", "   1 "))
    assert f0.rows[0].df_max_a == 21000.50
    # 7-column CTFFIND-style accepted
    f7 = AreTomoCtfFile.from_string(
        "1 21000.5 20000.2 12.0 0.0 0.15 4.5\n2 22000.0 21500.0 170.0 0.0 0.2 5.0\n"
    )
    assert f7.rows[0].df_hand is None


def test_ctf_file_rejects_duplicates_gaps_and_bad_lines():
    with pytest.raises(ValueError, match="contiguous unique"):
        AreTomoCtfFile.from_string("1 1 1 1 0 0 1 1\n1 2 2 2 0 0 1 1\n")  # duplicate
    with pytest.raises(ValueError, match="contiguous unique"):
        AreTomoCtfFile.from_string("1 1 1 1 0 0 1 1\n3 2 2 2 0 0 1 1\n")  # gap
    with pytest.raises(ValueError, match="7 or 8 columns"):
        AreTomoCtfFile.from_string("1 2 3\n")
    with pytest.raises(ValueError, match="no data rows"):
        AreTomoCtfFile.from_string("# only a header\n")


def test_ctf_file_row_count_must_match_series():
    ctf_file = AreTomoCtfFile.from_string("1 1 1 1 0 0 1 1\n2 2 2 2 0 0 1 1\n")
    with pytest.raises(ValueError, match="rows but the series has"):
        tiltctf_from_aretomo(ctf_file, torch.tensor([0.0, 30.0, 60.0]))


def test_ctf_file_writer_refuses_overwrite(tmp_path):
    f = AreTomoCtfFile.from_string("1 1 1 1 0 0 1 1\n")
    out = tmp_path / "x_CTF.txt"
    f.to_file(out)
    with pytest.raises(FileExistsError):
        f.to_file(out)


# --- Warp grids ---------------------------------------------------------------


def test_warp_grid_roundtrip(ts1_xml_path_module):
    series = load_warp_tiltseries(ts1_xml_path_module)
    ts = copy.deepcopy(series.ts)
    ctf = _random_ctf(ts.n_tilts, with_diag=False)
    ctf.voltage_kv = 200.0
    tiltctf_to_warp(ts, ctf)
    assert warp_has_tilt_ctf(ts)
    back = tiltctf_from_warp(ts)
    # grids are float32 in Warp
    torch.testing.assert_close(back.defocus_u_a, ctf.defocus_u_a, atol=0.5, rtol=0)
    torch.testing.assert_close(back.defocus_v_a, ctf.defocus_v_a, atol=0.5, rtol=0)
    torch.testing.assert_close(back.angle_deg, ctf.angle_deg, atol=1e-4, rtol=0)
    torch.testing.assert_close(back.phase_deg, ctf.phase_deg, atol=1e-3, rtol=0)
    assert back.voltage_kv == pytest.approx(200.0)


def test_warp_no_ctf_and_bad_dims(ts1_xml_path_module):
    from warpylib.cubic_grid import CubicGrid

    series = load_warp_tiltseries(ts1_xml_path_module)
    ts = copy.deepcopy(series.ts)
    ts.grid_ctf_defocus = CubicGrid((1, 1, 1))
    assert not warp_has_tilt_ctf(ts)
    assert tiltctf_from_warp(ts) is None

    ts2 = copy.deepcopy(series.ts)
    ts2.grid_ctf_defocus = CubicGrid((2, 2, ts2.n_tilts), values=torch.ones(4 * ts2.n_tilts))
    with pytest.raises(ValueError, match="expected \\(1, 1"):
        tiltctf_from_warp(ts2)


def test_xml_writer_ctf_roundtrip(tmp_path, ts1_xml_path_module):
    from cets_nonrigid.io.warp_xml import write_alignment_into_template

    series = load_warp_tiltseries(ts1_xml_path_module)
    ts = copy.deepcopy(series.ts)
    ctf = _random_ctf(ts.n_tilts, with_diag=False)
    ctf.voltage_kv, ctf.cs_mm, ctf.amplitude_contrast = 200.0, 2.6, 0.1
    tiltctf_to_warp(ts, ctf)
    out = tmp_path / "with_ctf.xml"
    write_alignment_into_template(series.xml_bytes, ts, out, with_ctf=True)

    re = load_warp_tiltseries(out)
    back = tiltctf_from_warp(re.ts)
    torch.testing.assert_close(back.defocus_u_a, ctf.defocus_u_a, atol=0.5, rtol=0)
    torch.testing.assert_close(back.angle_deg, ctf.angle_deg, atol=1e-4, rtol=0)
    torch.testing.assert_close(back.phase_deg, ctf.phase_deg, atol=1e-3, rtol=0)
    assert re.ts.ctf.voltage == pytest.approx(200.0)
    assert re.ts.ctf.cs == pytest.approx(2.6)
    assert re.ts.ctf.amplitude == pytest.approx(0.1)


_REAL_CTF_TXT = (
    "/hpc/projects/group.czii/utz.ermel/repos/arewarpo/testdata_runs/"
    "at3_24jul16a/out_ctf/24jul16a_Position_16_3_CTF.txt"
)


@pytest.mark.skipif(
    not __import__("pathlib").Path(_REAL_CTF_TXT).exists(),
    reason="genuine AreTomo3 _CTF.txt not generated (run AreTomo3 with -kV/-Cs)",
)
def test_parse_genuine_aretomo3_ctf_txt():
    """Parse golden against a real AreTomo3 (-kV 300 -Cs 2.7) _CTF.txt."""
    f = AreTomoCtfFile.from_file(_REAL_CTF_TXT)
    assert f.n_rows == 31  # full raw stack including any darks
    hands = {r.df_hand for r in f.rows}
    # dfHand is one consistent sign; the +1 normalization (CAreTomoMain.cpp:
    # 344-367) is CONDITIONAL on the -TiltAxis refine setting — the genuine
    # -kV 300 run on 24jul16a recorded -1, falsifying "always +1".
    assert hands <= {-1, 1} and len(hands) == 1
    for r in f.rows:
        assert r.df_max_a >= r.df_min_a  # DfMax >= DfMin by construction upstream
        assert 0 < r.df_max_a < 1e5
    # convertible into canonical form against the matching Warp XML tilt count
    angles = torch.linspace(45, -45, 31)  # descending like the real series
    ctf = tiltctf_from_aretomo(f, angles)
    assert ctf.n_tilts == 31
    assert torch.isfinite(ctf.defocus_u_a).all()


# --- G4: hand / depth-defocus golden ------------------------------------------


def _relion_depth_a(ts, points_a):
    """hand-free depth term: [R_relion (p_A - V_A/2)]_z per tilt, using the
    converter rotation mapping (xtilt=levelX, ytilt=-(angle+levelY),
    zrot=axisAngle). Translations cancel in getDepthOffset
    (tomogram.cpp:267-275: (P p).z - (P centre).z)."""
    from cets_nonrigid.models.relion_ts import RelionTomogramModel

    t = ts.n_tilts
    model = RelionTomogramModel(
        xtilt_deg=torch.full((t,), float(ts.level_angle_x), dtype=torch.float64),
        ytilt_deg=-(ts.angles.to(torch.float64) + float(ts.level_angle_y)),
        zrot_deg=ts.tilt_axis_angles.to(torch.float64),
        xshift_a=torch.zeros(t, dtype=torch.float64),
        yshift_a=torch.zeros(t, dtype=torch.float64),
        tomo_dims_px=(64, 64, 32),  # irrelevant: depth differences cancel centers
        image_dims_px=(128, 128),
        pixel_size_a=1.0,
    )
    r = model.rotations  # (T, 3, 3)
    centred = points_a.to(torch.float64) - ts.volume_dimensions_physical.to(torch.float64) / 2
    return torch.einsum("tij,nj->tni", r, centred)[..., 2]  # (T, N)


@pytest.mark.parametrize("inverted", [False, True], ids=["not_inverted", "inverted"])
def test_g4_hand_matches_warp_depth_defocus(ts1_xml_path_module, inverted):
    """Determine rlnTomoHand for the converter's Euler mapping by comparing
    warpylib's defocus channel (Warp ground truth) against RELION's
    dz = hand * depth formula (defocusSlope = 1)."""
    from warpylib.cubic_grid import CubicGrid
    from warpylib.linear_grid import LinearGrid4D
    from warpylib.tilt_series.positions import get_position_in_all_tilts_single

    from cets_nonrigid.conventions import RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED

    series = load_warp_tiltseries(ts1_xml_path_module)
    ts = copy.deepcopy(series.ts)
    ts.grid_movement_x = CubicGrid((1, 1, 1))
    ts.grid_movement_y = CubicGrid((1, 1, 1))
    ts.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))
    ts.grid_ctf_defocus = CubicGrid((1, 1, 1))  # zero: isolate the depth term
    ts.are_angles_inverted = inverted

    vol = ts.volume_dimensions_physical.to(torch.float64)
    pts = torch.tensor(RNG.uniform(0.15, 0.85, (24, 3))) * vol

    out = get_position_in_all_tilts_single(ts, pts.to(torch.float32))  # (N, T, 3)
    warp_defocus_a = out[..., 2].permute(1, 0).to(torch.float64) * 1e4  # um -> A

    depth = _relion_depth_a(ts, pts)  # (T, N), hand-free
    err_plus = (warp_defocus_a - depth).abs().max()
    err_minus = (warp_defocus_a + depth).abs().max()

    scale = depth.abs().max()
    if inverted:
        hand = -1 if err_minus < err_plus else 1
        best, other = min(err_minus, err_plus), max(err_minus, err_plus)
    else:
        hand = 1 if err_plus < err_minus else -1
        best, other = min(err_plus, err_minus), max(err_plus, err_minus)
    # decisively separated: float32 chain noise vs O(depth) signal
    assert best < 1e-2 * scale, f"neither hand matches: {err_plus=}, {err_minus=}"
    assert other > 0.5 * scale

    if not inverted:
        assert RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED == hand, (
            f"G4 determined hand={hand} for AreAnglesInverted=False; update "
            "conventions.RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED to match"
        )
    else:
        assert hand == -RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED
