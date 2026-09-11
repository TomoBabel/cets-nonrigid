"""Text-level post-checks for an emitted ``.aln`` (the model parser raises on
malformed local rows, so these run on the raw text).

Gates (pass / fail / not_evaluated):
  * ``sec_dense``   SEC strictly ascending in row order and, together with the
    ``# DarkFrame`` raw indices, covering every raw section exactly once
    (AreTomo3 applies rows positionally; SEC is the 1-based raw-sorted index)
  * ``rows_expected``  number of global rows == expected (when given)
  * ``tilt_dev``    max |TILT - source angle| <= tolerance (when source angles given)
  * ``local_rows_parsable``  no fixed-width overflow rows (e.g. ``81283.28-103507.92``)
  * ``patch_shift`` per-patch max |shift| <= threshold (when a threshold is given);
    patch nodes without particle support blow up while held-out metrics look fine
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cets_nonrigid.project.common import Gate, gate

_HEADER_INT = {
    "num_patches": re.compile(r"^#\s*NumPatches\s*=\s*(\d+)"),
    "thickness": re.compile(r"^#\s*Thickness\s*=\s*(\d+)"),
}
_RAWSIZE = re.compile(r"^#\s*RawSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)")
_DARK = re.compile(r"^#\s*DarkFrame\s*=\s*(\d+)\s+(\d+)\s+([-+\d.]+)")


class AlnCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    num_patches: int
    raw_size: tuple[int, int, int] | None
    thickness: int | None
    n_dark: int
    dark_idx: list[int]
    sec: list[int]
    tilts: list[float]
    sec_dense: bool
    tilt_ascending: bool
    max_abs_tilt_dev_deg: float | None = None
    local_rows_parsable: int = 0
    local_rows_unparsable: int = 0
    per_patch_max_shift_px: list[float] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if g.status == "fail"]


def _parse_float_tokens(line: str, n: int) -> list[float] | None:
    parts = line.split()
    if len(parts) != n:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def check_aln(
    text_or_path: str | Path,
    *,
    source_angles_deg: list[float] | None = None,
    expect_rows: int | None = None,
    max_patch_shift_px: float | None = None,
    tilt_dev_tol_deg: float = 2.0,
) -> AlnCheck:
    text = Path(text_or_path).read_text() if isinstance(text_or_path, Path) else str(text_or_path)
    raw_size = None
    num_patches = 0
    thickness = None
    n_dark = 0
    dark_idx: list[int] = []
    globals_: list[list[float]] = []
    local_lines: list[str] = []
    in_local = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "Local Alignment" in s:
                in_local = True
            elif s.startswith("# DarkFrame"):
                n_dark += 1
                md = _DARK.match(s)
                if md:
                    dark_idx.append(int(md.group(1)))
            m = _RAWSIZE.match(s)
            if m:
                raw_size = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for key, rx in _HEADER_INT.items():
                mm = rx.match(s)
                if mm:
                    if key == "num_patches":
                        num_patches = int(mm.group(1))
                    else:
                        thickness = int(mm.group(1))
            continue
        if in_local:
            local_lines.append(s)
        else:
            vals = _parse_float_tokens(s, 10)
            if vals is None:
                raise ValueError(f"unparsable global .aln row: {s!r}")
            globals_.append(vals)

    sec = [int(v[0]) for v in globals_]
    tilts = [v[9] for v in globals_]
    gates: list[Gate] = []
    ascending = all(b > a for a, b in pairwise(sec))
    covered = sorted([v - 1 for v in sec] + dark_idx)
    n_raw = raw_size[2] if raw_size is not None else len(sec) + len(dark_idx)
    sec_dense = ascending and covered == list(range(n_raw))
    gates.append(gate(
        "sec_dense", sec_dense, value=sec[:5],
        expected=f"ascending, SEC-1 + DarkFrame idx == 0..{n_raw - 1}",
        note="" if ascending else "SEC not ascending in row order",
    ))
    tilt_ascending = tilts == sorted(tilts)
    if expect_rows is not None:
        gates.append(gate("rows_expected", len(sec) == expect_rows, value=len(sec), expected=expect_rows))
    dev = None
    if source_angles_deg is not None:
        if len(source_angles_deg) == len(tilts):
            dev = max(abs(a - b) for a, b in zip(tilts, source_angles_deg))
            gates.append(gate("tilt_dev", dev <= tilt_dev_tol_deg, value=round(dev, 3),
                              expected=f"<= {tilt_dev_tol_deg} deg"))
        else:
            gates.append(gate("tilt_dev", None, value=len(source_angles_deg), expected=len(tilts),
                              note="source angle count differs from .aln rows"))

    parsable, unparsable = 0, 0
    per_patch: dict[int, float] = {}
    for s in local_lines:
        vals = _parse_float_tokens(s, 7)
        if vals is None:
            unparsable += 1
            continue
        parsable += 1
        p = int(vals[1])
        per_patch[p] = max(per_patch.get(p, 0.0), abs(vals[4]), abs(vals[5]))
    if num_patches > 0:
        gates.append(gate("local_rows_parsable", unparsable == 0, value=unparsable, expected=0,
                          note="fixed-width overflow rows" if unparsable else ""))
        expected_local = num_patches * len(sec)
        gates.append(gate("local_rows_count", parsable + unparsable == expected_local,
                          value=parsable + unparsable, expected=expected_local))
    per_patch_list = [per_patch.get(p, 0.0) for p in range(num_patches)]
    if num_patches > 0:
        if max_patch_shift_px is None:
            gates.append(gate("patch_shift", None, value=round(max(per_patch_list), 2) if per_patch_list else None,
                              expected="no threshold given"))
        else:
            worst = max(per_patch_list) if per_patch_list else 0.0
            gates.append(gate("patch_shift", worst <= max_patch_shift_px, value=round(worst, 2),
                              expected=f"<= {max_patch_shift_px} px"))
    return AlnCheck(
        n_rows=len(sec), num_patches=num_patches, raw_size=raw_size, thickness=thickness,
        n_dark=n_dark, dark_idx=dark_idx, sec=sec, tilts=tilts, sec_dense=sec_dense,
        tilt_ascending=tilt_ascending,
        max_abs_tilt_dev_deg=dev, local_rows_parsable=parsable, local_rows_unparsable=unparsable,
        per_patch_max_shift_px=per_patch_list, gates=gates,
    )
