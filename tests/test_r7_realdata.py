"""R7: real data — 24jul16a Warp processing -> w2r -> real extraction, then a
relion_tomo_align smoke on the bundle and r2w of align's own output.

Precondition (audited 2026-08-31): the align path (programs/align.*, the
whole motion/ module, optimisation_set/trajectory/particle_set/tomogram[_set])
is byte-identical between the installed 5.0.0 binary (5b1a6532) and the
210f68c8 source checkout — the single-tomogram --stack2d exception extends to
this smoke.

All tests skip cleanly without the warp_trial data or the RELION binaries.
"""

from pathlib import Path

import numpy as np
import pytest
import test_r3_relion_binary as rb
import torch

WARP_TRIAL = Path("/hpc/projects/group.czii/utz.ermel/warp_trial/24jul16a")
SERIES = "24jul16a_Position_16_3"
PIX = 1.54
VOL_PX = (4096, 4096, 2000)

_have_data = (WARP_TRIAL / "warp_tiltseries" / f"{SERIES}.xml").exists()
pytestmark = pytest.mark.skipif(
    not _have_data or rb.RELION_SUBTOMO is None,
    reason="warp_trial data or relion binaries unavailable",
)

RNG = np.random.default_rng(20260909)


@pytest.fixture(scope="module")
def template_xml(tmp_path_factory):
    from lxml import etree

    xml = WARP_TRIAL / "warp_tiltseries" / f"{SERIES}.xml"
    root = etree.fromstring(xml.read_bytes())
    root.set("ImageDimensionsAngstrom", f"{4096 * PIX}, {4096 * PIX}")
    root.set(
        "VolumeDimensionsAngstrom",
        f"{VOL_PX[0] * PIX}, {VOL_PX[1] * PIX}, {VOL_PX[2] * PIX}",
    )
    p = tmp_path_factory.mktemp("tmpl") / f"{SERIES}.xml"
    p.write_bytes(etree.tostring(root, xml_declaration=True, encoding="utf-8"))
    return p


@pytest.fixture(scope="module")
def tilt_image_list(tmp_path_factory, template_xml):
    """One corrected average image per EMITTED row (XML order, darks dropped),
    mapped from the tomostar's movie column — explicit list, never glob order."""
    import starfile

    from cets_nonrigid.io.warp_xml import load_warp_tiltseries

    tomostar = WARP_TRIAL / "tomostar" / f"{SERIES}.tomostar"
    if not tomostar.exists():
        pytest.skip("tomostar missing")
    df = next(iter(starfile.read(tomostar, always_dict=True).values()))
    avg_dir = WARP_TRIAL / "warp_frameseries" / "average"
    names_all = []
    for movie in df["wrpMovieName"]:
        stem = Path(str(movie)).name.replace(".eer", ".mrc")
        p = avg_dir / stem
        if not p.exists():
            pytest.skip(f"average image missing: {p}")
        names_all.append(str(p))

    series = load_warp_tiltseries(template_xml)
    if series.ts.n_tilts != len(names_all):
        pytest.skip("tomostar row count != XML tilt count")
    rows = [i for i in range(series.ts.n_tilts) if bool(series.ts.use_tilt[i])]
    out = tmp_path_factory.mktemp("imgs") / "tilt_images.txt"
    out.write_text("\n".join(names_all[i] for i in rows) + "\n")
    return out


@pytest.fixture(scope="module")
def picks():
    vol_a = torch.tensor([VOL_PX[0] * PIX, VOL_PX[1] * PIX, VOL_PX[2] * PIX], dtype=torch.float64)
    frac = torch.zeros(60, 3, dtype=torch.float64)
    frac[:, 0] = torch.tensor(RNG.uniform(0.25, 0.75, 60))
    frac[:, 1] = torch.tensor(RNG.uniform(0.25, 0.75, 60))
    frac[:, 2] = torch.tensor(RNG.uniform(0.3, 0.7, 60))
    return frac * vol_a


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory, template_xml, tilt_image_list, picks):
    from cets_nonrigid.convert_relion import warp_to_relion

    out = tmp_path_factory.mktemp("bundle")
    names = [f"{SERIES}/{i + 1}" for i in range(picks.shape[0])]
    r = warp_to_relion(
        template_xml, out / "bundle",
        pixel_size_a=PIX, tomo_name=SERIES,
        positions_eff_a=picks, particle_names=names,
        tilt_image_list=tilt_image_list,
    )
    return r


def test_w2r_real_bundle(real_bundle):
    r = real_bundle
    assert r.global_result.global_exact and not r.global_result.used_fallback
    assert r.lift.max_residual_px < 1e-3
    assert r.n_particles == 60
    # the real XML carries CTF grids -> a normal (CTF-bearing) bundle
    from cets_nonrigid.io.relion_star import read_tomograms_star

    tomo = read_tomograms_star(r.tomograms_star)[SERIES]
    assert tomo.ctf is not None
    assert float(tomo.ctf.defocus_u_a.min()) > 1000  # real defoci, not placeholders


def test_real_extraction_sanity(real_bundle, tmp_path):
    """Extraction runs on the real images and yields one 2D stack per pick."""
    import subprocess

    import mrcfile
    import starfile

    r = real_bundle
    out = tmp_path / "subtomo"
    cmd = [
        rb.RELION_SUBTOMO,
        "--p", str(r.particles_star), "--t", str(r.tomograms_star),
        "--mot", str(r.motion_star), "--o", str(out) + "/",
        "--b", "64", "--crop", "48", "--bin", "2",
        "--stack2d", "--j", "8",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    assert run.returncode == 0, f"relion_tomo_subtomo failed:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}"

    parts = starfile.read(out / "particles.star", always_dict=True)["particles"]
    assert len(parts) == 60
    assert "rlnTomoVisibleFrames" in parts.columns
    img0 = Path(str(parts["rlnImageName"].iloc[0]))
    with mrcfile.open(img0) as m:
        data = np.asarray(m.data)
    assert data.ndim in (2, 3)
    shape = data.shape if data.ndim == 3 else (1, *data.shape)
    assert shape[1] == 48 and shape[2] == 48
    assert 1 <= shape[0] <= 31  # visible frames of a real series
    assert np.isfinite(data).all() and float(np.abs(data).max()) > 0  # real signal


def _make_reference(tmp_path, box, pix):
    """Fabricated half-maps + mask (a soft Gaussian sphere): enough for the
    align SMOKE — the goal is that OUR files parse and the program runs."""
    import mrcfile

    zz, yy, xx = np.mgrid[0:box, 0:box, 0:box].astype(np.float64)
    c = (box - 1) / 2
    r2 = (xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2
    blob = np.exp(-r2 / (2 * (box / 8) ** 2)).astype(np.float32)
    mask = (np.exp(-r2 / (2 * (box / 5) ** 2)) > 0.05).astype(np.float32)
    paths = []
    for name, arr in (("ref1.mrc", blob), ("ref2.mrc", blob), ("mask.mrc", mask)):
        p = tmp_path / name
        with mrcfile.new(p) as f:
            f.set_data(arr)
            f.voxel_size = pix
        paths.append(p)
    return paths


def test_align_smoke_and_r2w_of_its_output(real_bundle, template_xml, tmp_path):
    """relion_tomo_align consumes our bundle (motion refinement smoke), and
    r2w consumes align's own output — the full polishing round trip on real
    metadata."""
    import subprocess

    r = real_bundle
    align_bin = str(Path(rb.RELION_SUBTOMO).parent / "relion_tomo_align")
    if not Path(align_bin).exists():
        pytest.skip("relion_tomo_align not found")

    box = 64
    ref1, ref2, mask = _make_reference(tmp_path, box, PIX * 2)
    out = tmp_path / "align"
    cmd = [
        align_bin,
        "--p", str(r.particles_star), "--t", str(r.tomograms_star),
        "--mot", str(r.motion_star),
        "--ref1", str(ref1), "--ref2", str(ref2), "--mask", str(mask),
        "--o", str(out) + "/",
        "--b", str(box),
        "--motion", "--r", "2", "--it", "100",
        "--j", "8",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    assert run.returncode == 0, f"relion_tomo_align failed:\n{run.stdout[-3000:]}\n{run.stderr[-2000:]}"
    for name in ("motion.star", "tomograms.star", "particles.star", "optimisation_set.star"):
        assert (out / name).exists(), f"align did not write {name}"

    # r2w of the real polishing output
    from cets_nonrigid.convert_relion import relion_to_warp

    res = relion_to_warp(
        template_xml, tmp_path / "roundtrip.xml",
        tomo_name=SERIES,
        optimisation_set=out / "optimisation_set.star",
        movement_grid=(4, 4),
        deactivate_unmatched=True,
    )
    assert res.fit.heldout_status == "evaluated"
    assert res.fit.rms_a_heldout is not None and np.isfinite(res.fit.rms_a_heldout)
    assert res.out_xml.exists()
