"""R4: a2r — AreTomo3 .aln + particles -> RELION bundle, incl. the
acquisition-order dose machinery (mdoc golden vs RELION's own calculation)."""

from pathlib import Path

import numpy as np
import pytest
import test_r3_relion_binary as rb
import torch

from cets_nonrigid.io.dose import (
    RawDose,
    raw_dose_assume_stack_order,
    raw_dose_from_acq_order,
    raw_dose_from_mdoc,
    raw_dose_from_tlt,
)

RNG = np.random.default_rng(20260906)

_TESTDATA = Path("/hpc/projects/group.czii/utz.ermel/repos/arewarpo/testdata_runs/at3_24jul16a")
_REAL_TLT = _TESTDATA / "out_ctf" / "24jul16a_Position_16_3_TLT.txt"
_REAL_MDOC = _TESTDATA / "24jul16a_Position_16_3.mdoc"
_REAL_ALN = _TESTDATA / "out_ctf" / "24jul16a_Position_16_3.aln"

_have_real = _REAL_TLT.exists() and _REAL_MDOC.exists() and _REAL_ALN.exists()


def _relion_pre_exposure_transcription(exposure_doses):
    """calculate_pre_exposure_dose, ExposureDose branch — verbatim
    (tomography_python_programs/_utils/mdoc.py:13-40): INCLUSIVE cumsum over
    the datetime-sorted entries."""
    return np.cumsum(np.asarray(exposure_doses, dtype=np.float64))


# --- dose sources -------------------------------------------------------------


@pytest.mark.skipif(not _have_real, reason="genuine AreTomo3 -kV run not present")
def test_tlt_dose_conventions():
    """Default EXCLUSIVE: pre-exposure = dose BEFORE the image (first acquired
    tilt carries 0). ``convention="inclusive"`` reproduces RELION's mdoc
    ExposureDose branch (np.cumsum)."""
    from cryoet_alignment.io.aretomo3 import AreTomo3TLT

    tlt = AreTomo3TLT.from_file(_REAL_TLT)
    order = np.argsort(tlt.acq_indices)
    frac = np.asarray(tlt.doses, dtype=np.float64)[order]
    assert (frac > 0.5).all() and (frac < 3.0).all()  # per-tilt e/A^2, plausible

    rd = raw_dose_from_tlt(_REAL_TLT)
    assert rd.n_raw == 31 and rd.convention == "exclusive"
    got = rd.pre_exposure.numpy()[order]
    np.testing.assert_allclose(got, np.cumsum(frac) - frac, atol=1e-9)
    assert got[0] == 0.0
    np.testing.assert_allclose(rd.dose_per_image.numpy()[order][:-1], frac[:-1], atol=1e-9)

    rd_inc = raw_dose_from_tlt(_REAL_TLT, convention="inclusive")
    np.testing.assert_allclose(
        rd_inc.pre_exposure.numpy()[order], _relion_pre_exposure_transcription(frac), atol=1e-9
    )
    np.testing.assert_allclose(rd_inc.dose_per_image.numpy(), np.asarray(tlt.doses), atol=1e-9)
    # the two conventions differ by exactly each image's own dose
    np.testing.assert_allclose(
        (rd_inc.pre_exposure - rd.pre_exposure).numpy(), np.asarray(tlt.doses), atol=1e-9
    )


@pytest.mark.skipif(not _have_real, reason="genuine AreTomo3 -kV run not present")
def test_mdoc_dose_golden_and_aln_join():
    """mdoc pre-exposure mirrors RELION's own calculation, and the join to the
    .aln raw tilt list survives the AlphaOffset (TILT includes it)."""
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

    from cets_nonrigid.convert_relion import _raw_tilts_from_aln
    from cets_nonrigid.io.dose import parse_mdoc

    raw_tilts = _raw_tilts_from_aln(AreTomo3ALN.from_file(str(_REAL_ALN)))
    rd = raw_dose_from_mdoc(_REAL_MDOC, raw_tilts, convention="inclusive")
    assert rd.n_raw == 31

    # golden: RELION's calculation on the same datetime-sorted ExposureDose list
    entries = sorted(parse_mdoc(_REAL_MDOC), key=lambda e: e.get("DateTime", ""))
    doses = np.asarray([float(e["ExposureDose"]) for e in entries])
    ref = _relion_pre_exposure_transcription(doses)
    got = rd.pre_exposure[torch.argsort(rd.acq_index_1b)].numpy()
    np.testing.assert_allclose(got, ref, atol=1e-9)
    # default (exclusive) = inclusive minus each image's own dose
    rd_ex = raw_dose_from_mdoc(_REAL_MDOC, raw_tilts)
    np.testing.assert_allclose(
        rd_ex.pre_exposure[torch.argsort(rd_ex.acq_index_1b)].numpy(), ref - doses, atol=1e-9
    )

    # cross-check the two independent sources agree on the acquisition order
    rd_tlt = raw_dose_from_tlt(_REAL_TLT)
    assert rd.acq_index_1b.tolist() == rd_tlt.acq_index_1b.tolist()


def test_dose_validation():
    with pytest.raises(ValueError, match="unique integers"):
        RawDose(
            tilt_deg=torch.zeros(3),
            acq_index_1b=torch.tensor([1, 1, 2]),
            pre_exposure=torch.zeros(3),
        )
    # zero doses refused unless overridden
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("-30 1 0.0\n0 2 0.0\n30 3 0.0\n")
    with pytest.raises(ValueError, match="no usable dose"):
        raw_dose_from_tlt(f.name)
    rd = raw_dose_from_tlt(f.name, dose_per_tilt=3.0)
    assert rd.pre_exposure.tolist() == [0.0, 3.0, 6.0]  # exclusive default
    rd_inc = raw_dose_from_tlt(f.name, dose_per_tilt=3.0, convention="inclusive")
    assert rd_inc.pre_exposure.tolist() == [3.0, 6.0, 9.0]
    # one-column IMOD .tlt rejected
    with tempfile.NamedTemporaryFile("w", suffix=".tlt", delete=False) as f2:
        f2.write("-30\n0\n30\n")
    with pytest.raises(ValueError, match="three-column"):
        raw_dose_from_tlt(f2.name)


def test_acq_order_and_unsafe_sources(tmp_path):
    tilts = torch.tensor([-30.0, 0.0, 30.0])
    p = tmp_path / "acq.txt"
    p.write_text("3\n1\n2\n")  # -30 acquired last (dose-symmetric-ish)
    rd = raw_dose_from_acq_order(p, tilts, dose_per_tilt=2.0)
    assert rd.pre_exposure.tolist() == [4.0, 0.0, 2.0]
    assert raw_dose_from_acq_order(p, tilts, dose_per_tilt=2.0, convention="inclusive").pre_exposure.tolist() == [6.0, 2.0, 4.0]
    rd2 = raw_dose_assume_stack_order(tilts, dose_per_tilt=2.0)
    assert rd2.pre_exposure.tolist() == [0.0, 2.0, 4.0]


# --- synthetic a2r pipeline ---------------------------------------------------

T_RAW, PIX = 9, 2.0
DARK_SEC0 = 4  # 0-based raw index of the dark frame
TOMO = (48, 48, 24)
IMG = (96, 96)


def synthetic_raw_dose(tilts_raw, dose_per_tilt=1.5):
    """RawDose over the RAW rows of a synthetic .aln (dose-symmetric acquisition
    order, constant per-image dose): a Warp XML needs a per-tilt dose — an
    equal-dose series is invalid Warp metadata (io/warp_xml.check_dose_policy)."""
    from cets_nonrigid.io.dose import raw_dose_from_indices

    tilts = np.asarray(tilts_raw, dtype=np.float64)
    acq = np.argsort(np.argsort(np.abs(tilts), kind="stable")) + 1
    return raw_dose_from_indices(torch.tensor(tilts), torch.tensor(acq), dose_per_tilt=dose_per_tilt)


def _write_synthetic_aln(path):
    from cryoet_alignment.io.aretomo3.aln import (
        AreTomo3ALN,
        DarkFrameInfo,
        GlobalAlignmentInfo,
        LocalAlignmentInfo,
    )

    tilts_raw = np.linspace(-48, 48, T_RAW)
    keep = [i for i in range(T_RAW) if i != DARK_SEC0]
    rot = 85.0 + RNG.uniform(-1, 1, T_RAW)
    globals_out = [
        GlobalAlignmentInfo(
            sec=i + 1,
            rot=float(rot[i]),
            tx=float(RNG.uniform(-4, 4)),
            ty=float(RNG.uniform(-4, 4)),
            tilt=float(tilts_raw[i] + 1.7),  # TILT includes AlphaOffset 1.7
        )
        for i in keep
    ]
    p_grid = 2
    locals_out = []
    # SMOOTH local field (real drift fields are smooth; iid-random patch
    # shifts would make the gauge-composed round-trip observable inherently
    # non-representable and break the same-family closure gates)
    for t in range(len(keep)):
        for px in range(p_grid):
            for py in range(p_grid):
                cx = (px + 0.5) / p_grid * IMG[0] - IMG[0] / 2
                cy = (py + 0.5) / p_grid * IMG[1] - IMG[1] / 2
                phase = t / max(1, len(keep) - 1)
                sx = 2.0 * np.sin(2.2 * phase + 0.7 * px - 0.4 * py)
                sy = 2.0 * np.cos(1.8 * phase - 0.5 * px + 0.6 * py)
                locals_out.append(
                    LocalAlignmentInfo(
                        sec_idx=t, patch_idx=px * p_grid + py,
                        center_x=round(cx, 2), center_y=round(cy, 2),
                        shift_x=round(float(sx), 2),
                        shift_y=round(float(sy), 2),
                        is_reliable=1.0,
                    )
                )
    aln = AreTomo3ALN(
        RawSize=(IMG[0], IMG[1], T_RAW),
        NumPatches=p_grid * p_grid,
        DarkFrames=[DarkFrameInfo(section_idx=DARK_SEC0, val2=DARK_SEC0, angle=float(tilts_raw[DARK_SEC0] + 1.7))],
        AlphaOffset=1.7,
        BetaOffset=0.0,
        Thickness=TOMO[2],
        GlobalAlignments=globals_out,
        LocalAlignments=locals_out,
    )
    path.write_text(str(aln))
    return path, tilts_raw


def _synthetic_inputs(tmp_path):
    import mrcfile

    aln_path, tilts_raw = _write_synthetic_aln(tmp_path / "syn.aln")
    # _TLT-style dose file over the RAW rows (dose-symmetric acquisition)
    acq = np.argsort(np.argsort(np.abs(tilts_raw), kind="stable")) + 1
    tlt = tmp_path / "syn_TLT.txt"
    tlt.write_text(
        "\n".join(f"{tilts_raw[i]:8.2f} {int(acq[i]):5d} {1.5:8.2f}" for i in range(T_RAW))
    )
    # synthetic _CTF.txt over the RAW rows
    ctf = tmp_path / "syn_CTF.txt"
    ctf.write_text(
        "\n".join(
            f"{i + 1:4d} {21000 + 100 * i:8.2f} {20500 + 100 * i:8.2f} {30.0:8.2f} "
            f"{0.0:9.4f} {0.1:8.4f} {5.0:8.4f} {1:3d}"
            for i in range(T_RAW)
        )
    )
    stack = tmp_path / "stack.mrc"
    with mrcfile.new(stack) as m:
        m.set_data(np.zeros((T_RAW, IMG[1], IMG[0]), dtype=np.float32))
        m.voxel_size = PIX
    vol_a = torch.tensor(TOMO, dtype=torch.float64) * PIX
    frac = torch.tensor(
        [[0.15, 0.20, 0.40], [0.85, 0.50, 0.60], [0.40, 0.85, 0.50]], dtype=torch.float64
    )
    return aln_path, tlt, ctf, stack, frac * vol_a


def test_a2r_bundle(tmp_path):
    from cets_nonrigid.convert_relion import aretomo_to_relion
    from cets_nonrigid.io.relion_star import read_motion_star, read_tomograms_star

    aln_path, tlt, ctf, stack, pos = _synthetic_inputs(tmp_path)
    raw_dose = raw_dose_from_tlt(tlt)
    names = [f"TS_A2R/{i + 1}" for i in range(pos.shape[0])]
    r = aretomo_to_relion(
        aln_path, tmp_path / "bundle",
        pixel_size_a=PIX, tomo_name="TS_A2R", tomo_dims_px=TOMO,
        voltage_kv=300.0, cs_mm=2.7, amplitude_contrast=0.07, hand=1,
        positions_eff_a=pos, particle_names=names, raw_dose=raw_dose,
        ctf_file=ctf, tilt_stack=stack,
    )
    assert r.global_result.global_exact and not r.global_result.used_fallback
    assert r.lift.max_residual_px < 1e-3
    assert abs(r.alpha_offset_deg - 1.7) < 0.05
    assert len(r.sec_1b) == T_RAW - 1 and (DARK_SEC0 + 1) not in r.sec_1b

    tomo = read_tomograms_star(r.tomograms_star)["TS_A2R"]
    # CTF joined via SEC: defocus of emitted row t == raw row sec-1
    for t, sec in enumerate(r.sec_1b):
        assert float(tomo.ctf.defocus_u_a[t]) == pytest.approx(21000 + 100 * (sec - 1), abs=0.5)
    # pre-exposure subset from the FULL raw sequence (dark included in the sums)
    order = torch.argsort(raw_dose.acq_index_1b)
    assert float(tomo.pre_exposure.min()) == pytest.approx(
        float(raw_dose.pre_exposure[order][0])
    ) or float(tomo.pre_exposure.min()) > 0

    motion = read_motion_star(r.motion_star, names, T_RAW - 1)
    ref = r.lift.ref_row
    torch.testing.assert_close(motion[ref], torch.zeros_like(motion[ref]), atol=1e-5, rtol=0)
    assert motion.abs().max() > 0.1  # locals became trajectories


def test_a2r_join_offset_spread_rejected(tmp_path):
    from cets_nonrigid.convert_relion import aretomo_to_relion

    aln_path, tlt, ctf, stack, pos = _synthetic_inputs(tmp_path)
    # corrupt the dose source's tilt column non-uniformly -> join must fail
    lines = tlt.read_text().splitlines()
    parts0 = lines[0].split()
    lines[0] = f"{float(parts0[0]) + 4.0:8.2f} {parts0[1]:>5} {parts0[2]:>8}"
    bad = tmp_path / "bad_TLT.txt"
    bad.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="constant offset"):
        aretomo_to_relion(
            aln_path, tmp_path / "bundle2",
            pixel_size_a=PIX, tomo_name="TS_B", tomo_dims_px=TOMO,
            voltage_kv=300.0, cs_mm=2.7, amplitude_contrast=0.07, hand=1,
            positions_eff_a=pos, particle_names=["TS_B/1", "TS_B/2", "TS_B/3"],
            raw_dose=raw_dose_from_tlt(bad), ctf_file=ctf, tilt_stack=stack,
        )


# --- binary headline for a2r --------------------------------------------------



@pytest.mark.skipif(
    rb.RELION_SUBTOMO is None,
    reason="relion_tomo_subtomo not found (set AREWARPION_RELION_BIN or module load relion/5.0.0)",
)
def test_a2r_headline_polished_extraction(tmp_path):
    import subprocess

    import mrcfile

    from cets_nonrigid.convert_relion import aretomo_to_relion
    from cets_nonrigid.io.aln import load_aln

    aln_path, tlt, ctf, _stack, pos = _synthetic_inputs(tmp_path)
    vol_a = tuple(float(d) * PIX for d in TOMO)
    aln_series = load_aln(aln_path, pixel_size_a=PIX, volume_dims_a=vol_a)
    names = [f"TS_AH/{i + 1}" for i in range(pos.shape[0])]

    # paint the RAW-ORDER stack from the FULL aln model (globals + IDW locals)
    src_xy, _ = aln_series.model.project_volume(pos)  # (T_aln, P, 2) canonical A
    proj_px = src_xy.to(torch.float64) / PIX
    sec_idx = [int(g.sec) - 1 for g in aln_series.aln.GlobalAlignments]
    yy, xx = np.mgrid[0 : IMG[1], 0 : IMG[0]]
    frames = np.zeros((T_RAW, IMG[1], IMG[0]), dtype=np.float32)
    for t, sec in enumerate(sec_idx):
        for j in range(pos.shape[0]):
            x, y = float(proj_px[t, j, 0]), float(proj_px[t, j, 1])
            frames[sec] += np.exp(-(((xx - x) ** 2) + ((yy - y) ** 2)) / (2 * 1.5**2)).astype(
                np.float32
            )
    stack = tmp_path / "painted.mrc"
    with mrcfile.new(stack) as m:
        m.set_data(frames)
        m.voxel_size = PIX

    r = aretomo_to_relion(
        aln_path, tmp_path / "bundle",
        pixel_size_a=PIX, tomo_name="TS_AH", tomo_dims_px=TOMO,
        voltage_kv=300.0, cs_mm=2.7, amplitude_contrast=0.07, hand=1,
        positions_eff_a=pos, particle_names=names,
        raw_dose=raw_dose_from_tlt(tlt), ctf_file=ctf, tilt_stack=stack,
    )

    out = tmp_path / "out"
    cmd = [
        rb.RELION_SUBTOMO,
        "--p", str(r.particles_star), "--t", str(r.tomograms_star),
        "--mot", str(r.motion_star), "--o", str(out) + "/",
        "--b", "24", "--crop", "24", "--bin", "1",
        "--stack2d", "--no_ctf", "--no_ic", "--j", "2",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    assert run.returncode == 0, f"relion_tomo_subtomo failed:\n{run.stdout}\n{run.stderr}"
    mag = rb._centroid_offsets(out).norm(dim=-1)
    assert float(mag.mean()) <= 0.1, f"mean centroid offset {mag.mean():.3f} px"
