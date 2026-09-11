"""WP2 binary acceptance: a project written by the a2r CLI (root-relative
paths, N@<stack> micrograph names) extracts centred particles with real
RELION run from the project root."""

import subprocess
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
import test_r3_relion_binary as rb
import test_r4_a2r as helpers
import torch



@pytest.mark.skipif(
    rb.RELION_SUBTOMO is None,
    reason="relion_tomo_subtomo not found (set CETS_NONRIGID_RELION_BIN or module load relion/5.0.0)",
)
def test_a2r_project_extracts_centered_with_stack_refs(tmp_path, monkeypatch):
    from cets_nonrigid.io.aln import load_aln

    aln_path, tlt, ctf, _stack, pos = helpers._synthetic_inputs(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    for src, dst in ((aln_path, "TS_A.aln"), (tlt, "TS_A_TLT.txt"), (ctf, "TS_A_CTF.txt")):
        (run / dst).write_text(Path(src).read_text())
    vol_a = tuple(float(d) * helpers.PIX for d in helpers.TOMO)
    aln_series = load_aln(run / "TS_A.aln", pixel_size_a=helpers.PIX, volume_dims_a=vol_a)

    # paint the RAW-ORDER stack from the FULL aln model (globals + IDW locals)
    src_xy, _ = aln_series.model.project_volume(pos)
    proj_px = src_xy.to(torch.float64) / helpers.PIX
    sec_idx = [int(g.sec) - 1 for g in aln_series.aln.GlobalAlignments]
    yy, xx = np.mgrid[0 : helpers.IMG[1], 0 : helpers.IMG[0]]
    frames = np.zeros((helpers.T_RAW, helpers.IMG[1], helpers.IMG[0]), dtype=np.float32)
    for t, sec in enumerate(sec_idx):
        for j in range(pos.shape[0]):
            x, y = float(proj_px[t, j, 0]), float(proj_px[t, j, 1])
            frames[sec] += np.exp(-(((xx - x) ** 2) + ((yy - y) ** 2)) / (2 * 1.5**2)).astype(np.float32)
    with mrcfile.new(run / "TS_A.mrc") as m:
        m.set_data(frames)
        m.voxel_size = helpers.PIX
    picks = tmp_path / "picks.txt"
    picks.write_text("\n".join(" ".join(f"{float(v):.3f}" for v in row) for row in pos.tolist()))

    root = tmp_path / "relion"
    from test_wp1_cli import invoke, OPTICS
    res = invoke(tmp_path, "aretomo3-to-relion", run / "TS_A.aln", root,
        source_options={"tomo_size_px":list(helpers.TOMO), "particles":str(picks), "defocus_handedness":1, **OPTICS})
    assert res.exit_code == 0, res.output
    ts = starfile.read(root / "tilt_series" / "TS_A.star")
    assert all(n.endswith("@tilt_series/TS_A.mrcs") for n in ts["rlnMicrographName"])
    assert (root / "tilt_series" / "TS_A.mrcs").is_file()

    out = root / "Subtomo"
    cmd = [
        rb.RELION_SUBTOMO, "--p", "particles.star", "--t", "tomograms.star", "--mot", "motion.star",
        "--o", "Subtomo/", "--b", "24", "--crop", "24", "--bin", "1", "--stack2d", "--no_ctf", "--no_ic", "--j", "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False, cwd=root)
    assert proc.returncode == 0, f"relion_tomo_subtomo failed:\n{proc.stdout}\n{proc.stderr}"
    monkeypatch.chdir(root)  # RELION wrote root-relative rlnImageName paths
    mag = rb._centroid_offsets(out).norm(dim=-1)
    assert float(mag.mean()) <= 0.1, f"mean centroid offset {mag.mean():.3f} px"
