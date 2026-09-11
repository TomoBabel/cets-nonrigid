"""R3 binary format acceptance: a hand-constructed bundle through the real
``relion_tomo_subtomo`` (no converter involved — star IO + known parameters).

Synthetic impulse stacks: Gaussian impulses painted at the torch model's
predicted projections of known 3D points; the bundle must extract them
CENTERED (geometry decoupled from CTF via --no_ctf; CTF/handedness is G11's
job in R4).

Executable discovery (documented contract): $CETS_NONRIGID_RELION_BIN (a
directory containing the relion binaries) is probed first, then $PATH
(populate via ``module load relion/5.0.0``); tests skip cleanly when absent.
The installed 5.0.0-commit-5b1a65 matches the 210f68c8 source checkout on
this path for single-tomogram --stack2d use only.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from cets_nonrigid.ctf import TiltCtf
from cets_nonrigid.io.relion_star import (
    RelionTomogramData,
    write_motion_star,
    write_optimisation_set,
    write_particles_star,
    write_tomograms_star,
)
from cets_nonrigid.models.relion_ts import RelionTomogramModel

RNG = np.random.default_rng(20260904)


def _find_relion() -> str | None:
    env_dir = os.environ.get("CETS_NONRIGID_RELION_BIN")
    if env_dir and (Path(env_dir) / "relion_tomo_subtomo").exists():
        return str(Path(env_dir) / "relion_tomo_subtomo")
    return shutil.which("relion_tomo_subtomo")


RELION_SUBTOMO = _find_relion()
pytestmark = pytest.mark.skipif(
    RELION_SUBTOMO is None,
    reason="relion_tomo_subtomo not found (set CETS_NONRIGID_RELION_BIN or module load relion/5.0.0)",
)

T, PIX = 8, 2.0
TOMO_DIMS = (48, 48, 24)
IMG_DIMS = (96, 96)
BOX = 24
SIGMA = 1.5


def _model():
    return RelionTomogramModel(
        xtilt_deg=torch.zeros(T, dtype=torch.float64),
        ytilt_deg=torch.linspace(-52.5, 52.5, T, dtype=torch.float64),
        zrot_deg=torch.full((T,), 85.0, dtype=torch.float64),
        xshift_a=torch.tensor(RNG.uniform(-10, 10, T)),
        yshift_a=torch.tensor(RNG.uniform(-10, 10, T)),
        tomo_dims_px=TOMO_DIMS,
        image_dims_px=IMG_DIMS,
        pixel_size_a=PIX,
    )


def _particles():
    # well inside the volume AND far enough apart that no neighbour's impulse
    # can enter another particle's BOX (separations >= ~20 px at all tilts)
    vol_a = torch.tensor(TOMO_DIMS, dtype=torch.float64) * PIX
    frac = torch.tensor(
        [[0.15, 0.20, 0.40], [0.85, 0.50, 0.60], [0.40, 0.85, 0.50]],
        dtype=torch.float64,
    )
    return frac * vol_a


def _paint_stack(dir_: Path, positions_px: torch.Tensor) -> list:
    """positions_px (T, P, 2) -> per-tilt MRCs with Gaussian impulses."""
    import mrcfile

    names = []
    yy, xx = np.mgrid[0 : IMG_DIMS[1], 0 : IMG_DIMS[0]]
    for f in range(T):
        img = np.zeros((IMG_DIMS[1], IMG_DIMS[0]), dtype=np.float32)
        for j in range(positions_px.shape[1]):
            x, y = float(positions_px[f, j, 0]), float(positions_px[f, j, 1])
            img += np.exp(-(((xx - x) ** 2) + ((yy - y) ** 2)) / (2 * SIGMA**2)).astype(np.float32)
        path = dir_ / f"tilt_{f:03d}.mrc"
        with mrcfile.new(path) as m:
            m.set_data(img)
            m.voxel_size = PIX
        names.append(str(path))
    return names


def _write_bundle(out: Path, model, positions_a, motion_a=None):
    tilt_dir = out / "tilts"
    tilt_dir.mkdir(parents=True)
    pos = positions_a[None, :, :].expand(T, -1, -1).clone()
    if motion_a is not None:
        pos = pos + motion_a
    proj_px = torch.empty(T, pos.shape[1], 2, dtype=torch.float64)
    for f in range(T):
        r = model.projection_matrices[f, :3, :3]
        t = model.projection_matrices[f, :3, 3]
        proj_px[f] = (torch.einsum("ij,pj->pi", r, pos[f] / PIX) + t)[:, :2]
    names = _paint_stack(tilt_dir, proj_px)

    p_count = positions_a.shape[0]
    ctf = TiltCtf(  # finite placeholders; extraction runs --no_ctf (geometry-only)
        defocus_u_a=torch.full((T,), 20000.0, dtype=torch.float64),
        defocus_v_a=torch.full((T,), 20000.0, dtype=torch.float64),
        angle_deg=torch.zeros(T, dtype=torch.float64),
        phase_deg=torch.zeros(T, dtype=torch.float64),
    )
    tomo = RelionTomogramData(
        name="TS_SYN",
        voltage_kv=300.0,
        cs_mm=2.7,
        amplitude_contrast=0.07,
        hand=1,
        pixel_size_a=PIX,
        tomo_dims_px=TOMO_DIMS,
        image_dims_px=IMG_DIMS,
        xtilt_deg=model._angles[0],
        ytilt_deg=model._angles[1],
        zrot_deg=model._angles[2],
        xshift_a=model._angles[3],
        yshift_a=model._angles[4],
        pre_exposure=torch.arange(T, dtype=torch.float64) * 3.0,
        nominal_stage_angle_deg=-model._angles[1],  # any consistent ordering key
        ctf=ctf,
        micrograph_names=names,
    )
    write_tomograms_star(out, tomo)
    centre_a = torch.tensor([d / 2.0 for d in TOMO_DIMS], dtype=torch.float64) * PIX
    p_names = [f"TS_SYN/{i + 1}" for i in range(p_count)]
    write_particles_star(
        out / "particles.star",
        tomo_name="TS_SYN",
        particle_names=p_names,
        centered_coords_a=positions_a - centre_a,
        voltage_kv=300.0,
        cs_mm=2.7,
        amplitude_contrast=0.07,
        pixel_size_a=PIX,
    )
    mot = None
    if motion_a is not None:
        mot = write_motion_star(out / "motion.star", p_names, motion_a)
    write_optimisation_set(
        out / "optimisation_set.star",
        particles=str(out / "particles.star"),
        tomograms=str(out / "tomograms.star"),
        trajectories=str(mot) if mot else None,
    )
    return out


def _run_subtomo(bundle: Path, out: Path, with_motion: bool):
    cmd = [
        RELION_SUBTOMO,
        "--p", str(bundle / "particles.star"),
        "--t", str(bundle / "tomograms.star"),
        "--o", str(out) + "/",
        "--b", str(BOX), "--crop", str(BOX), "--bin", "1",
        "--stack2d", "--no_ctf", "--no_ic", "--j", "2",
    ]
    if with_motion:
        cmd += ["--mot", str(bundle / "motion.star")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, f"relion_tomo_subtomo failed:\n{r.stdout}\n{r.stderr}"
    return out


def _centroid_offsets(out: Path) -> torch.Tensor:
    """Per (particle, frame) centroid offset from the box center, px."""
    import mrcfile
    import starfile

    parts = starfile.read(out / "particles.star", always_dict=True)["particles"]
    offsets = []
    win = 4  # windowed centroid around the peak: neighbouring particles' tails
    # inside the small box would otherwise pollute a global centroid
    for img_name in parts["rlnImageName"]:
        path = Path(str(img_name))  # relative names resolve against the run cwd
        with mrcfile.open(path) as m:
            data = np.asarray(m.data, dtype=np.float64)
        if data.ndim == 2:
            data = data[None]
        for frame in data:
            iy, ix = np.unravel_index(np.argmax(frame), frame.shape)
            y0, y1 = max(0, iy - win), min(frame.shape[0], iy + win + 1)
            x0, x1 = max(0, ix - win), min(frame.shape[1], ix + win + 1)
            v = np.clip(frame[y0:y1, x0:x1], 0, None)
            total = v.sum()
            assert total > 0
            ys, xs = np.mgrid[y0:y1, x0:x1]
            cx = (v * xs).sum() / total
            cy = (v * ys).sum() / total
            offsets.append([cx - BOX / 2, cy - BOX / 2])
    return torch.tensor(offsets, dtype=torch.float64)


def test_impulse_extraction_is_centered(tmp_path):
    model = _model()
    bundle = _write_bundle(tmp_path / "bundle", model, _particles())
    out = _run_subtomo(bundle, tmp_path / "out", with_motion=False)
    off = _centroid_offsets(out)
    assert off.shape[0] >= 3 * T // 2  # visible frames exist
    mag = off.norm(dim=-1)
    assert float(mag.mean()) <= 0.1, f"mean centroid offset {mag.mean():.3f} px"
    assert float(mag.quantile(0.95)) <= 0.2, f"p95 centroid offset {mag.quantile(0.95):.3f} px"


def test_impulse_extraction_with_trajectories(tmp_path):
    model = _model()
    positions = _particles()
    p = positions.shape[0]
    # smooth per-particle drifts up to ~3 px, zero at the lowest-dose frame (row 0)
    ramp = torch.linspace(0, 1, T, dtype=torch.float64)[:, None, None]
    direction = torch.tensor(RNG.uniform(-1, 1, (1, p, 3)))  # up to ~6 A = 3 px
    motion_a = ramp * direction * 6.0
    motion_a[0] = 0.0

    bundle = _write_bundle(tmp_path / "bundle", model, positions, motion_a=motion_a)
    out = _run_subtomo(bundle, tmp_path / "out", with_motion=True)
    mag = _centroid_offsets(out).norm(dim=-1)
    assert float(mag.mean()) <= 0.1, f"mean centroid offset {mag.mean():.3f} px (with motion.star)"

    # sanity contrast: ignoring the trajectories must leave the impulses off-center
    out2 = _run_subtomo(bundle, tmp_path / "out_nomot", with_motion=False)
    mag2 = _centroid_offsets(out2).norm(dim=-1)
    assert float(mag2.max()) > 0.5, "trajectories had no effect on extraction?"
