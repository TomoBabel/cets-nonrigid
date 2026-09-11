"""M6 contract test: the VERBATIM production MotionCor kernel vs the Python
model.

The harness (tools/mckernel) runs the mGCorrect3D kernel extracted verbatim
from AreTomo3 at build time. The input frame encodes each pixel's own index,
so the kernel's output reveals exactly which source pixel it sampled for
every output pixel — pinning the evaluation coordinate (output pixel index,
corner origin, no half-pixel), the weighting/cutoff/bad-patch gate, the
truncation-based NN sampling, the upsampling semantics, and the out-of-range
scramble. This is the test that freezes the .mcaln map semantics.
"""

import subprocess
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cets_nonrigid.models.aretomo_motion import AretomoMotionModel

HARNESS = Path(os.environ.get("CETS_MCKERNEL_HARNESS", Path(__file__).parent.parent / ".scratch" / "native" / "mckernel" / "mckernel_harness"))

pytestmark = pytest.mark.skipif(
    not HARNESS.exists() or not torch.cuda.is_available(),
    reason="mckernel harness or GPU unavailable",
)

RNG = np.random.default_rng(2026)


def _run_kernel(nx, ny, centers, shifts, bad, up, frame_up, tmp_path):
    np_in = np.concatenate(
        [
            centers.astype(np.float32).ravel(),
            shifts.astype(np.float32).ravel(),
            bad.astype(np.float32).ravel(),
            frame_up.astype(np.float32).ravel(),
        ]
    )
    fin = tmp_path / "in.bin"
    fout = tmp_path / "out.bin"
    np_in.tofile(fin)
    subprocess.run(
        [str(HARNESS), str(nx), str(ny), str(len(centers)), str(up), str(fin), str(fout)],
        check=True,
        capture_output=True,
    )
    pad_x = (nx // 2 + 1) * 2
    return np.fromfile(fout, dtype=np.float32).reshape(ny, pad_x)


def _predict(nx, ny, centers, shifts, bad, up):
    """Python-model prediction of the sampled source index per output pixel."""
    model = AretomoMotionModel(
        global_shifts_px=torch.zeros(1, 2),
        patch_centers_px=torch.tensor(centers, dtype=torch.float64),
        patch_shifts_px=torch.tensor(shifts, dtype=torch.float64)[None],
        patch_valid=torch.tensor(~bad.astype(bool))[None],
        frame_size_px=(nx, ny),
        pixel_size_a=1.0,
    )
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    pts = torch.tensor(np.stack([xs.ravel(), ys.ravel()], axis=-1), dtype=torch.float64)
    s = model.local_field_px(pts, mode="compat")[0].numpy()  # (N, 2) float32

    up_pad_x = ((nx * up) // 2 + 1) * 2
    size_x, size_y = nx * up, ny * up
    src_x = ((xs.ravel() - s[:, 0]) * up).astype(np.int64)  # C float->int truncation
    src_y = ((ys.ravel() - s[:, 1]) * up).astype(np.int64)

    oob = (src_x < 0) | (src_y < 0) | (src_x >= size_x) | (src_y >= size_y)
    sx = np.where(src_x < 0, -src_x, src_x)
    sy = np.where(src_y < 0, -src_y, src_y)
    src_x = np.where(oob, (811 * sx) % size_x, src_x)
    src_y = np.where(oob, (811 * sy) % size_y, src_y)
    return (src_y * up_pad_x + src_x).reshape(ny, nx)


def _index_frame(nx, ny, up):
    up_pad_x = ((nx * up) // 2 + 1) * 2
    frame = np.arange(up_pad_x * ny * up, dtype=np.float64).reshape(ny * up, up_pad_x)
    assert frame.max() < 2**24  # exact in float32
    return frame


@pytest.mark.parametrize("up", [1, 2])
def test_kernel_contract(tmp_path, up):
    nx, ny = 96, 80
    centers = np.stack(
        np.meshgrid((np.arange(3) + 0.5) * nx / 3, (np.arange(3) + 0.5) * ny / 3, indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    shifts = RNG.uniform(-4, 4, centers.shape)
    bad = np.zeros(len(centers))
    bad[4] = 1.0  # kernel must skip this patch

    frame = _index_frame(nx, ny, up)
    out = _run_kernel(nx, ny, centers, shifts, bad, up, frame, tmp_path)

    predicted = _predict(nx, ny, centers, shifts, bad, up)
    got = out[:, :nx].astype(np.int64)

    mismatch = got != predicted
    # float32 kernel arithmetic can land a coordinate on the other side of an
    # integer boundary in rare cases; require essentially perfect agreement.
    assert mismatch.mean() < 0.001, f"{mismatch.sum()} of {mismatch.size} pixels differ"


def test_kernel_contract_no_good_patch(tmp_path):
    nx, ny = 64, 64
    centers = np.array([[32.0, 32.0]])
    shifts = np.array([[3.0, -2.0]])
    bad = np.ones(1)  # all patches bad -> zero shift everywhere

    frame = _index_frame(nx, ny, 1)
    out = _run_kernel(nx, ny, centers, shifts, bad, 1, frame, tmp_path)

    up_pad_x = (nx // 2 + 1) * 2
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    identity = ys * up_pad_x + xs
    assert (out[:, :nx].astype(np.int64) == identity).all()
