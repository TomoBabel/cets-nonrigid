"""RELION per-micrograph motion star (Micrograph::write/read format).

Blocks (src/micrograph_model.cpp:183-293, each preceded by ``# version
50001``; empty tables are silently absent):

  data_general            list: rlnImageSizeX/Y/Z, rlnMicrographMovieName,
                          [gain/defects], rlnMicrographBinning,
                          rlnMicrographOriginalPixelSize, rlnMicrographDoseRate,
                          rlnMicrographPreExposure, rlnVoltage,
                          rlnMicrographStartFrame (1-indexed),
                          [rlnEER*], rlnMotionModelVersion (1 = third-order
                          polynomial, 0 = none)
  data_global_shift       loop: rlnMicrographFrameNumber (1-indexed, ALL
                          rlnImageSizeZ frames), rlnMicrographShiftX/Y
                          (unbinned movie px; NOT_OBSERVED = -9999 outside the
                          modeled range; the shift is the CORRECTION, zero at
                          the first used frame)
  data_local_motion_model loop: rlnMotionModelCoeffsIdx 0-35 +
                          rlnMotionModelCoeff (X 0-17, Y 18-35; exactly 36
                          rows or the block is absent)

``data_hot_pixels``/``data_local_shift`` are diagnostic; the writer omits
them and the reader ignores them (upstream never reads local_shift back).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
import torch

from cets_nonrigid.conventions import RELION_NOT_OBSERVED, RELION_POLY_N_COEFFS

_F64 = torch.float64
VERSION_COMMENT = "# version 50001"


@dataclass
class RelionMicrographMotion:
    image_size_px: tuple  # (X, Y) unbinned movie dims
    n_frames: int  # rlnImageSizeZ
    global_shifts_px: torch.Tensor  # (F, 2), NOT_OBSERVED sentinels included
    poly_coeffs: torch.Tensor | None  # (36,) or None (model version 0)
    pixel_size_a: float | None  # rlnMicrographOriginalPixelSize
    start_frame: int = 1  # rlnMicrographStartFrame (1-indexed)
    movie_name: str = ""
    binning: float = 1.0
    dose_rate: float | None = None  # e/A^2/frame
    pre_exposure: float | None = None  # e/A^2
    voltage_kv: float | None = None
    eer_upsampling: int | None = None
    eer_grouping: int | None = None

    def to_model(self):
        from cets_nonrigid.models.relion_motion import RelionMicrographMotionModel

        if self.pixel_size_a is None:
            raise ValueError(
                "the motion star carries no rlnMicrographOriginalPixelSize — supply one"
            )
        return RelionMicrographMotionModel(
            global_shifts_px=self.global_shifts_px,
            poly_coeffs=self.poly_coeffs,
            image_size_px=self.image_size_px,
            pixel_size_a=self.pixel_size_a,
            start_frame=self.start_frame,
            binning=self.binning,
        )


def write_micrograph_motion_star(path: str | Path, m: RelionMicrographMotion) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    if m.global_shifts_px.shape != (m.n_frames, 2):
        raise ValueError(
            f"global shifts {tuple(m.global_shifts_px.shape)} vs rlnImageSizeZ {m.n_frames} "
            "— ALL frames must have a row (sentinel -9999 outside the modeled range)"
        )
    if m.poly_coeffs is not None and m.poly_coeffs.shape != (RELION_POLY_N_COEFFS,):
        raise ValueError(f"poly_coeffs must be ({RELION_POLY_N_COEFFS},) or None")

    general = {
        "rlnImageSizeX": int(m.image_size_px[0]),
        "rlnImageSizeY": int(m.image_size_px[1]),
        "rlnImageSizeZ": int(m.n_frames),
        "rlnMicrographMovieName": m.movie_name or "unknown.eer",
        "rlnMicrographBinning": float(m.binning),
    }
    if m.pixel_size_a is not None:
        general["rlnMicrographOriginalPixelSize"] = float(m.pixel_size_a)
    if m.dose_rate is not None:
        general["rlnMicrographDoseRate"] = float(m.dose_rate)
    if m.pre_exposure is not None:
        general["rlnMicrographPreExposure"] = float(m.pre_exposure)
    if m.voltage_kv is not None:
        general["rlnVoltage"] = float(m.voltage_kv)
    general["rlnMicrographStartFrame"] = int(m.start_frame)
    if m.eer_upsampling is not None:
        general["rlnEERUpsampling"] = int(m.eer_upsampling)
    if m.eer_grouping is not None:
        general["rlnEERGrouping"] = int(m.eer_grouping)
    general["rlnMotionModelVersion"] = 1 if m.poly_coeffs is not None else 0

    g = m.global_shifts_px.to(_F64).numpy()
    blocks = {
        "general": general,
        "global_shift": pd.DataFrame(
            {
                "rlnMicrographFrameNumber": np.arange(1, m.n_frames + 1),
                "rlnMicrographShiftX": g[:, 0],
                "rlnMicrographShiftY": g[:, 1],
            }
        ),
    }
    if m.poly_coeffs is not None:
        blocks["local_motion_model"] = pd.DataFrame(
            {
                "rlnMotionModelCoeffsIdx": np.arange(RELION_POLY_N_COEFFS),
                "rlnMotionModelCoeff": m.poly_coeffs.to(_F64).numpy(),
            }
        )
    starfile.write(blocks, path)
    path.write_text(f"{VERSION_COMMENT}\n" + path.read_text())
    return path


def read_micrograph_motion_star(path: str | Path) -> RelionMicrographMotion:
    blocks = starfile.read(Path(path), always_dict=True)
    if "general" not in blocks:
        raise ValueError(f"{path}: no data_general block")
    gen = blocks["general"]
    if not isinstance(gen, (dict, pd.Series)):
        gen = gen.iloc[0]

    def need(key):
        if key not in gen:
            raise ValueError(f"{path}: missing {key}")  # hard error upstream too
        return gen[key]

    n_frames = int(need("rlnImageSizeZ"))
    if "global_shift" not in blocks:
        raise ValueError(f"{path}: no data_global_shift block")
    gdf = blocks["global_shift"]
    shifts = torch.full((n_frames, 2), float(RELION_NOT_OBSERVED), dtype=_F64)
    frames = np.asarray(gdf["rlnMicrographFrameNumber"], dtype=np.int64)
    if (frames < 1).any() or (frames > n_frames).any():
        raise ValueError(f"{path}: rlnMicrographFrameNumber outside 1..{n_frames}")
    # the frame-number column is authoritative and 1-indexed (:533-534)
    shifts[frames - 1, 0] = torch.tensor(np.asarray(gdf["rlnMicrographShiftX"], dtype=np.float64))
    shifts[frames - 1, 1] = torch.tensor(np.asarray(gdf["rlnMicrographShiftY"], dtype=np.float64))

    coeffs = None
    version = int(gen.get("rlnMotionModelVersion", 0))
    if "local_motion_model" in blocks:
        ldf = blocks["local_motion_model"]
        if len(ldf) != RELION_POLY_N_COEFFS:
            raise ValueError(
                f"{path}: {len(ldf)} motion-model coefficients, expected exactly "
                f"{RELION_POLY_N_COEFFS} (micrograph_model.cpp:123-126)"
            )
        idx = np.asarray(ldf["rlnMotionModelCoeffsIdx"], dtype=np.int64)
        vals = np.asarray(ldf["rlnMotionModelCoeff"], dtype=np.float64)
        c = np.zeros(RELION_POLY_N_COEFFS)
        if sorted(idx.tolist()) != list(range(RELION_POLY_N_COEFFS)):
            raise ValueError(f"{path}: motion-model coefficient indices are not 0..35")
        c[idx] = vals
        coeffs = torch.tensor(c, dtype=_F64)
    elif version == 1:
        raise ValueError(f"{path}: rlnMotionModelVersion 1 but no data_local_motion_model block")

    return RelionMicrographMotion(
        image_size_px=(int(need("rlnImageSizeX")), int(need("rlnImageSizeY"))),
        n_frames=n_frames,
        global_shifts_px=shifts,
        poly_coeffs=coeffs,
        pixel_size_a=(
            float(gen["rlnMicrographOriginalPixelSize"])
            if "rlnMicrographOriginalPixelSize" in gen
            else None
        ),
        start_frame=int(gen.get("rlnMicrographStartFrame", 1)),
        movie_name=str(need("rlnMicrographMovieName")),
        binning=float(gen.get("rlnMicrographBinning", 1.0)),
        dose_rate=float(gen["rlnMicrographDoseRate"]) if "rlnMicrographDoseRate" in gen else None,
        pre_exposure=(
            float(gen["rlnMicrographPreExposure"]) if "rlnMicrographPreExposure" in gen else None
        ),
        voltage_kv=float(gen["rlnVoltage"]) if "rlnVoltage" in gen else None,
        eer_upsampling=int(gen["rlnEERUpsampling"]) if "rlnEERUpsampling" in gen else None,
        eer_grouping=int(gen["rlnEERGrouping"]) if "rlnEERGrouping" in gen else None,
    )
