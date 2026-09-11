"""`.mcaln` v1 — AreTomo3 2D motion alignment text format (frozen; see
docs/mcaln_format.md for the pinned semantics)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from cets_nonrigid.models.aretomo_motion import AretomoMotionModel

MAGIC = "# AreTomo3 MotionAlign 1.0"


class McAlnFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    integrated_index: int
    source_start: int
    source_count: int
    included: bool
    aligned_index: int  # -1 when excluded


class McAln(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_frame_count: int
    integrated_frame_count: int
    aligned_frame_count: int
    alignment_image_size_px: tuple[int, int]
    alignment_pixel_size_a: float
    input_to_alignment_scale_xy: tuple[float, float] = (1.0, 1.0)
    patches: tuple[int, int]
    fm_ref: int  # aligned-frame index space
    frames: list[McAlnFrame]
    global_shifts: list[tuple[int, float, float]]  # (aligned_index, sx, sy)
    # per patch: (patch_id, [(aligned_index, cx, cy, sx, sy, valid), ...])
    local_shifts: list[tuple[int, list[tuple[int, float, float, float, float, bool]]]]

    # ------------------------------------------------------------------

    def validate_consistency(self) -> None:
        f_aln = self.aligned_frame_count
        p_count = self.patches[0] * self.patches[1]

        if self.integrated_frame_count != len(self.frames):
            raise ValueError("frame table length != integrated_frame_count")
        aligned = [fr.aligned_index for fr in self.frames if fr.included]
        if sorted(aligned) != list(range(f_aln)):
            raise ValueError("aligned_index values of included frames must be 0..F-1")
        for fr in self.frames:
            if not fr.included and fr.aligned_index != -1:
                raise ValueError(f"excluded frame {fr.integrated_index} must have aligned_index -1")
        if not (0 <= self.fm_ref < f_aln):
            raise ValueError(f"fmRef {self.fm_ref} outside aligned range")

        g_idx = [g[0] for g in self.global_shifts]
        if sorted(g_idx) != list(range(f_aln)):
            raise ValueError("globalShift must cover each aligned frame exactly once")
        if len(self.local_shifts) != p_count:
            raise ValueError(f"expected {p_count} localShift blocks, got {len(self.local_shifts)}")
        for pid, rows in self.local_shifts:
            if sorted(r[0] for r in rows) != list(range(f_aln)):
                raise ValueError(f"localShift patch {pid} must cover each aligned frame once")
        for vals in (
            [v for g in self.global_shifts for v in g[1:]],
            [v for _, rows in self.local_shifts for r in rows for v in r[1:5]],
        ):
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("non-finite shift value")

    # ------------------------------------------------------------------

    def to_string(self) -> str:
        self.validate_consistency()
        out = [MAGIC, "setting"]
        out.append(f"   raw_frame_count: {self.raw_frame_count}")
        out.append(f"   integrated_frame_count: {self.integrated_frame_count}")
        out.append(f"   aligned_frame_count: {self.aligned_frame_count}")
        out.append(
            f"   alignment_image_size_px: {self.alignment_image_size_px[0]} "
            f"{self.alignment_image_size_px[1]}"
        )
        out.append(f"   alignment_pixel_size_A: {self.alignment_pixel_size_a:.6g}")
        out.append(
            f"   input_to_alignment_scale_xy: {self.input_to_alignment_scale_xy[0]:.6g} "
            f"{self.input_to_alignment_scale_xy[1]:.6g}"
        )
        out.append(f"   patches: {self.patches[0]} {self.patches[1]}")
        out.append(f"   fmRef: {self.fm_ref}")
        out.append("   frame_index_base: 0")
        out.append("frameTable")
        for fr in self.frames:
            out.append(
                f"   {fr.integrated_index:4d} {fr.source_start:6d} {fr.source_count:5d} "
                f"{int(fr.included):d} {fr.aligned_index:4d}"
            )
        out.append("globalShift")
        for f, sx, sy in self.global_shifts:
            out.append(f"   {f:4d} {sx:10.4f} {sy:10.4f}")
        for pid, rows in self.local_shifts:
            out.append("localShift")
            out.append(f"   patchID: {pid}")
            for f, cx, cy, sx, sy, valid in rows:
                out.append(
                    f"   {f:4d} {cx:9.2f} {cy:9.2f} {sx:9.3f} {sy:9.3f} {int(valid):d}"
                )
        return "\n".join(out) + "\n"

    @classmethod
    def from_string(cls, text: str) -> McAln:
        lines = text.splitlines()
        if not lines or lines[0].strip() != MAGIC:
            raise ValueError(f"missing/unsupported magic line (expected {MAGIC!r})")

        setting: dict = {}
        frames: list[McAlnFrame] = []
        global_shifts: list[tuple[int, float, float]] = []
        local_blocks: list[tuple[int, list]] = []
        section = None
        for raw in lines[1:]:
            line = raw.strip()
            if not line:
                continue
            if line in ("setting", "frameTable", "globalShift", "localShift"):
                section = line
                if line == "localShift":
                    local_blocks.append((-1, []))
                continue
            if section == "setting":
                key, _, val = line.partition(":")
                setting[key.strip()] = val.strip()
            elif section == "frameTable":
                p = line.split()
                frames.append(
                    McAlnFrame(
                        integrated_index=int(p[0]),
                        source_start=int(p[1]),
                        source_count=int(p[2]),
                        included=bool(int(p[3])),
                        aligned_index=int(p[4]),
                    )
                )
            elif section == "globalShift":
                p = line.split()
                global_shifts.append((int(p[0]), float(p[1]), float(p[2])))
            elif section == "localShift":
                if line.startswith("patchID"):
                    local_blocks[-1] = (int(line.partition(":")[2]), local_blocks[-1][1])
                else:
                    p = line.split()
                    local_blocks[-1][1].append(
                        (int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                         bool(int(p[5])))
                    )
            else:
                raise ValueError(f"content outside any section: {line!r}")

        if setting.get("frame_index_base", "0") != "0":
            raise ValueError("only frame_index_base 0 is supported")

        size = setting["alignment_image_size_px"].split()
        scale = setting.get("input_to_alignment_scale_xy", "1 1").split()
        patches = setting["patches"].split()
        obj = cls(
            raw_frame_count=int(setting["raw_frame_count"]),
            integrated_frame_count=int(setting["integrated_frame_count"]),
            aligned_frame_count=int(setting["aligned_frame_count"]),
            alignment_image_size_px=(int(size[0]), int(size[1])),
            alignment_pixel_size_a=float(setting["alignment_pixel_size_A"]),
            input_to_alignment_scale_xy=(float(scale[0]), float(scale[1])),
            patches=(int(patches[0]), int(patches[1])),
            fm_ref=int(setting["fmRef"]),
            frames=frames,
            global_shifts=global_shifts,
            local_shifts=local_blocks,
        )
        obj.validate_consistency()
        return obj

    @classmethod
    def from_file(cls, path: str | Path) -> McAln:
        return cls.from_string(Path(path).read_text())

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(self.to_string())

    # ------------------------------------------------------------------

    def to_model(self, idw_mode: str = "stable") -> AretomoMotionModel:
        """Build the motion model (aligned-frame indexing)."""
        f_aln = self.aligned_frame_count
        p_count = self.patches[0] * self.patches[1]

        glob = torch.zeros(f_aln, 2, dtype=torch.float64)
        for f, sx, sy in self.global_shifts:
            glob[f] = torch.tensor([sx, sy])

        centers = torch.zeros(p_count, 2, dtype=torch.float64)
        shifts = torch.zeros(f_aln, p_count, 2, dtype=torch.float64)
        valid = torch.zeros(f_aln, p_count, dtype=torch.bool)
        for slot, (pid, rows) in enumerate(self.local_shifts):
            for f, cx, cy, sx, sy, ok in rows:
                centers[slot] = torch.tensor([cx, cy])
                shifts[f, slot] = torch.tensor([sx, sy])
                valid[f, slot] = ok

        return AretomoMotionModel(
            global_shifts_px=glob,
            patch_centers_px=centers,
            patch_shifts_px=shifts,
            patch_valid=valid,
            frame_size_px=self.alignment_image_size_px,
            pixel_size_a=self.alignment_pixel_size_a,
            idw_mode=idw_mode,  # type: ignore[arg-type]
        )
