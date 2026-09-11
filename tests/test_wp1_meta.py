"""WP1: acquisition-metadata discovery + resolution for AreTomo3 runs, the
_TLT.txt companion, the .aln checker and the AreTomo3 output directory."""

from pathlib import Path

import numpy as np
import pytest
import torch

from cets_nonrigid.io.aln_check import check_aln
from cets_nonrigid.io.dose import raw_dose_from_pre_exposure
from cets_nonrigid.io.tlt import tlt_from_aln
from cets_nonrigid.meta import Discovered, MetaConflictError, resolve_series
from cets_nonrigid.meta.aretomo_run import discover_aretomo_series

_RUN = Path("/hpc/projects/group.czii/utz.ermel/repos/arewarpo/testdata_runs/at3_24jul16a")
_ALN = _RUN / "out_ctf" / "24jul16a_Position_16_3.aln"
_have_real = _ALN.exists() and (_RUN / "24jul16a_Position_16_3.mdoc").exists()
real = pytest.mark.skipif(not _have_real, reason="genuine AreTomo3 run not present")


# --- discovery on the real run ------------------------------------------------


@real
def test_discovery_finds_every_stem_adjacent_source():
    d = discover_aretomo_series(_ALN, mdoc_dir=_RUN)
    provs = {k: [str(c.provenance) for c in v] for k, v in d.candidates.items()}
    assert any("mrc#header" in p for p in provs["pixel_size_a"])
    assert any("mdoc#PixelSpacing" in p for p in provs["pixel_size_a"])
    assert d.first("pixel_size_a") == pytest.approx(1.54)
    assert d.first("image_dims_px") == (4096, 4096) and d.first("n_raw_sections") == 31
    assert d.first("tomo_dims_px") == (4096, 4096, 2000)  # _Vol.mrc 1024x500x1024 @ bin 4, FlipVol 0
    assert d.first("voltage_kv") == 300.0
    assert [str(c.provenance) for c in d.candidates["acq_order_1b"]] == [
        "file:24jul16a_Position_16_3_TLT.txt (tlt)", "file:24jul16a_Position_16_3.mdoc (mdoc)"]
    assert d.candidates["acq_order_1b"][0].value == d.candidates["acq_order_1b"][1].value
    names = d.first("tilt_image_names")
    assert len(names) == 31 and names[0].startswith("24jul16a_Position_16_3_031_-45.00")
    assert Path(d.first("ctf_path")).name == "24jul16a_Position_16_3_CTF.txt"
    assert Path(d.first("stack_path")).name == "24jul16a_Position_16_3.mrc"
    assert "amplitude_contrast" not in d.candidates and "defocus_hand" not in d.candidates


@real
def test_resolution_dose_guard_and_cli_precedence():
    d = discover_aretomo_series(_ALN, mdoc_dir=_RUN)
    m = resolve_series(d, {"cs_mm": 2.7, "amplitude_contrast": 0.07, "defocus_hand": -1})
    # the mdoc/TLT doses are tilt-attenuated (post-specimen): refused, reported
    assert m.dose_per_tilt is None and m.raw_dose is None
    assert "dose_per_tilt" in m.defaulted()
    assert any("looks post-specimen" in w for w in m.warnings)
    assert m.get_provenance("pixel_size_a").kind == "file"
    assert m.get_provenance("cs_mm").kind == "cli"

    m2 = resolve_series(d, {"cs_mm": 2.7, "amplitude_contrast": 0.07, "dose_per_tilt": 3.87})
    assert m2.raw_dose is not None and m2.raw_dose.convention == "exclusive"
    order = torch.argsort(m2.raw_dose.acq_index_1b)
    np.testing.assert_allclose(m2.raw_dose.pre_exposure[order].numpy(), 3.87 * np.arange(31), atol=1e-9)

    m3 = resolve_series(d, {"cs_mm": 2.7, "amplitude_contrast": 0.07}, dose_from="tlt")
    assert m3.raw_dose is not None
    assert float(m3.raw_dose.dose_per_image.min()) > 1.0  # the file's own per-image doses

    # CLI wins over a discovered file value, with a warning
    m4 = resolve_series(d, {"pixel_size_a": 1.6, "cs_mm": 2.7, "amplitude_contrast": 0.07})
    assert m4.pixel_size_a == 1.6 and m4.get_provenance("pixel_size_a").kind == "cli"
    assert any("--pix" in w and "overrides" in w for w in m4.warnings)


def test_resolution_conflict_between_files_is_an_error():
    d = Discovered("syn", "syn.aln")
    d.add("pixel_size_a", 1.54, "file", "stack#header")
    d.add("pixel_size_a", 1.60, "file", "mdoc#PixelSpacing")
    with pytest.raises(MetaConflictError, match="pixel_size_a"):
        resolve_series(d, {})
    m = resolve_series(d, {"pixel_size_a": 1.54})  # CLI settles it
    assert m.pixel_size_a == 1.54 and any("overrides" in w for w in m.warnings)


def test_resolution_agreeing_files_pass():
    d = Discovered("syn", "syn.aln")
    d.add("pixel_size_a", 1.54, "file", "a")
    d.add("pixel_size_a", 1.540001, "file", "b")
    d.add("image_dims_px", (4096, 4096), "file", "a")
    d.add("image_dims_px", [4096, 4096], "file", "b")
    m = resolve_series(d, {})
    assert m.pixel_size_a == 1.54 and m.get_provenance("pixel_size_a").source == "a"


# --- _TLT.txt companion -------------------------------------------------------


@real
def test_tlt_companion_from_real_aln():
    from cryoet_alignment.io.aretomo3 import AreTomo3ALN, AreTomo3TLT

    aln = AreTomo3ALN.from_file(str(_ALN))
    d = discover_aretomo_series(_ALN, mdoc_dir=_RUN)
    m = resolve_series(d, {"dose_per_tilt": 3.87})
    tlt = tlt_from_aln(aln, m.raw_dose)
    assert tlt.n_rows == 31
    # refined TILT (AlphaOffset -2.20 inside), unlike AreTomo3's own _TLT.txt (stage angles)
    own = AreTomo3TLT.from_file(str(_RUN / "out_ctf" / "24jul16a_Position_16_3_TLT.txt"))
    assert tlt.tilts[0] == pytest.approx(own.tilts[0] - 2.20, abs=0.011)
    assert tlt.acq_indices == own.acq_indices
    assert all(dz == pytest.approx(3.87) for dz in tlt.doses)
    assert str(tlt).splitlines()[0] == "  -47.21    31      3.87"


def test_tlt_from_synthetic_aln_with_darks(tmp_path):
    import test_r4_a2r as helpers
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

    aln_path, tilts_raw = helpers._write_synthetic_aln(tmp_path / "syn.aln")
    aln = AreTomo3ALN.from_file(str(aln_path))
    tlt = tlt_from_aln(aln, None)
    assert tlt.n_rows == helpers.T_RAW and not tlt.has_acq_index
    # dark angle re-inserted at its raw index
    assert tlt.tilts[helpers.DARK_SEC0] == pytest.approx(tilts_raw[helpers.DARK_SEC0] + 1.7, abs=0.01)
    pre = torch.tensor([3.0 * k for k in range(helpers.T_RAW)])
    rd = raw_dose_from_pre_exposure(torch.tensor(tilts_raw + 1.7), pre)
    assert rd.convention == "exclusive" and rd.acq_index_1b.tolist() == list(range(1, helpers.T_RAW + 1))
    tlt2 = tlt_from_aln(aln, rd)
    assert tlt2.doses == pytest.approx([3.0] * helpers.T_RAW)


def test_tlt_refuses_non_ascending_sec():
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN, GlobalAlignmentInfo

    with pytest.raises(ValueError, match="SEC"):  # cryoet-alignment >= 0.3 refuses it at construction
        aln = AreTomo3ALN(
            RawSize=(10, 10, 2), NumPatches=0, DarkFrames=[], AlphaOffset=0.0, BetaOffset=0.0,
            GlobalAlignments=[
                GlobalAlignmentInfo(sec=2, rot=0.0, tx=0.0, ty=0.0, tilt=-10.0),
                GlobalAlignmentInfo(sec=1, rot=0.0, tx=0.0, ty=0.0, tilt=10.0),
            ],
        )
        tlt_from_aln(aln, None)


# --- .aln checker -------------------------------------------------------------


@real
def test_check_aln_real_passes_and_reports_patches():
    c = check_aln(_ALN, max_patch_shift_px=100.0)
    assert c.n_rows == 31 and c.num_patches == 16 and c.sec_dense and c.local_rows_unparsable == 0
    assert {g.name: g.status for g in c.gates} == {
        "sec_dense": "pass", "local_rows_parsable": "pass", "local_rows_count": "pass", "patch_shift": "pass"}
    c2 = check_aln(_ALN)
    assert next(g for g in c2.gates if g.name == "patch_shift").status == "not_evaluated"


def test_check_aln_overflow_and_permuted_sec():
    text = (
        "# AreTomo Alignment / Priims bprmMn\n# RawSize = 100 100 2\n# NumPatches = 1\n"
        "# AlphaOffset = 0.00\n# BetaOffset = 0.00\n# Thickness = 50\n"
        "# SEC ROT GMAG TX TY SMEAN SFIT SCALE BASE TILT\n"
        "    2   -95.0    1.00000     -2.829     17.637     1.00     1.00     1.00     0.00   -10.00\n"
        "    1   -95.0    1.00000      3.497     33.431     1.00     1.00     1.00     0.00    10.00\n"
        "# Local Alignment\n"
        "    0    0   -10.00   -20.00      1.50      2.50  1.00\n"
        "    1    0   -10.00   -20.00 81283.28-103507.92  1.00\n"
    )
    c = check_aln(text, source_angles_deg=[-10.0, 10.0], max_patch_shift_px=100.0)
    assert not c.sec_dense and c.local_rows_unparsable == 1 and c.local_rows_parsable == 1
    status = {g.name: g.status for g in c.gates}
    assert status["sec_dense"] == "fail" and status["local_rows_parsable"] == "fail"
    assert status["tilt_dev"] == "pass" and status["local_rows_count"] == "pass"
    assert c.per_patch_max_shift_px == [2.5]


# --- AreTomo3 output directory ------------------------------------------------


@real
def test_aretomo_dir_finalize_gates_and_hint(tmp_path):
    import shutil

    from cryoet_alignment.io.aretomo3 import AreTomo3ALN

    from cets_nonrigid.io.aln import write_aln
    from cets_nonrigid.project.aretomo import AretomoDir

    root = tmp_path / "at3"
    root.mkdir()
    aln = AreTomo3ALN.from_file(str(_ALN))
    aln_path = root / "TS.aln"
    check = write_aln(aln_path, aln)
    shutil.copy(_RUN / "out_ctf" / "24jul16a_Position_16_3_CTF.txt", root / "TS_CTF.txt")
    d = discover_aretomo_series(_ALN, mdoc_dir=_RUN)
    m = resolve_series(d, {"dose_per_tilt": 3.87, "cs_mm": 2.7, "amplitude_contrast": 0.07})
    out = AretomoDir(root).finalize_series(
        "TS", aln, aln_path=aln_path, aln_check=check, tlt=tlt_from_aln(aln, m.raw_dose),
        ctf_path=root / "TS_CTF.txt", stack=Path(m.stack_path),
        hint_kwargs={"pixel_size_a": m.pixel_size_a, "voltage_kv": m.voltage_kv, "cs_mm": 2.7,
                     "amplitude_contrast": 0.07, "vol_z_px": 2000},
    )
    assert sorted(p.name for p in root.iterdir()) == ["TS.aln", "TS.mrc", "TS_CTF.txt", "TS_TLT.txt"]
    assert (root / "TS.mrc").is_symlink()
    assert not [g for g in out.gates if g.status == "fail"]
    names = {g.name for g in out.gates}
    assert {"ctf_rows", "ctf_no_blank_lines", "ctf_leading_space", "stack_sections", "stack_dims",
            "stack_voxel", "path_length"} <= names
    assert "-VolZ 2000 -AtBin <bin>" in out.hint and "-PixSize 1.54 -kV 300" in out.hint and "-CorrCTF 1" in out.hint
    # a stale, different .rawtlt is flagged
    (root / "TS.rawtlt").write_text("\n".join(f"{v:.2f}" for v in range(31)) + "\n")
    out2 = AretomoDir(root).finalize_series(
        "TS", aln, aln_path=aln_path, aln_check=check, tlt=tlt_from_aln(aln, m.raw_dose),
        ctf_path=None, stack=None,
    )
    assert next(g for g in out2.gates if g.name == "no_conflicting_rawtlt").status == "fail"


def test_aretomo_dir_prepare_refuses_existing(tmp_path):
    from cets_nonrigid.project.aretomo import AretomoDir

    root = tmp_path / "at3"
    root.mkdir()
    (root / "TS.aln").write_text("x")
    with pytest.raises(FileExistsError, match="--overwrite"):
        AretomoDir(root).prepare("TS", overwrite=False)
    AretomoDir(root).prepare("TS", overwrite=True)
    assert not (root / "TS.aln").exists()
