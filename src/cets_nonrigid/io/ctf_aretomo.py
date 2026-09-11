"""AreTomo3 ``<name>_CTF.txt`` reader/writer.

Format (CSaveCtfResults.cpp:69-99, verified): header comment lines starting
with ``#``, then one row per RAW tilt (the tilt-angle-ascending stack
INCLUDING dark frames — CTF is estimated before dark removal,
CAreTomoMain.cpp:92-130), ``"%4d %8.2f %8.2f %8.2f %9.4f %8.4f %8.4f %3d"``:

  col 1  micrograph number (1-based row index into the sorted raw stack)
  col 2  defocus 1 = DfMax  [Angstrom]   (>= col 3 by construction upstream)
  col 3  defocus 2 = DfMin  [Angstrom]
  col 4  azimuth of astigmatism [degrees]  (direction of DfMax; NOT wrapped)
  col 5  additional phase shift [RADIANS]
  col 6  cross-correlation score
  col 7  fit resolution limit [Angstrom]
  col 8  dfHand (+1 or -1; the +1 normalization via a 180-degree tilt-axis
         rotation, CAreTomoMain.cpp:344-367, is CONDITIONAL on the -TiltAxis
         refine setting — a genuine -kV 300 run on 24jul16a recorded -1)

CTFFIND4-style 7-column files (no dfHand) are accepted (df_hand = None).

Row keying: the file carries ONLY the micrograph number — rows are joined to
other representations by normalizing the contiguous number column by its
minimum (AreTomo3's own loader accepts 0- or 1-based the same way,
CLoadCtfResults.cpp:84-94) and treating the result as the ordinal into the raw
ascending-tilt-sorted table including darks. Duplicates, gaps, and
out-of-range indices are rejected.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

_HEADER = (
    "# Columns: #1 micrograph number; #2 - defocus 1 [A]; #3 - defocus 2; "
    "#4 - azimuth of astigmatism;\n"
    "#5 - additional phase shift [radian]; #6 - cross correlation;\n"
    "#7 - spacing (in Angstroms) up to which CTF rings were fit successfully; #8 - dfHand\n"
)


class AreTomoCtfRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    micrograph: int  # as read; normalized ordinal = micrograph - min(column)
    df_max_a: float
    df_min_a: float
    azimuth_deg: float
    phase_rad: float
    score: float
    res_a: float
    df_hand: int | None = None


class AreTomoCtfFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[AreTomoCtfRow]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def model_post_init(self, context, /) -> None:
        if not self.rows:
            raise ValueError("_CTF.txt contains no data rows")
        nums = [r.micrograph for r in self.rows]
        base = min(nums)
        ordinals = [n - base for n in nums]
        if sorted(ordinals) != list(range(len(nums))):
            raise ValueError(
                "_CTF.txt micrograph-number column is not a contiguous unique "
                f"sequence (after normalizing by its minimum {base}): {nums}"
            )
        # store rows in ordinal order so row i == raw-stack ordinal i
        order = sorted(range(len(nums)), key=lambda i: ordinals[i])
        object.__setattr__(self, "rows", [self.rows[i] for i in order])
        import math

        for r in self.rows:
            for name in ("df_max_a", "df_min_a", "azimuth_deg", "phase_rad", "score", "res_a"):
                if not math.isfinite(getattr(r, name)):
                    raise ValueError(f"non-finite {name} in _CTF.txt row {r.micrograph}")

    # -- parsing ------------------------------------------------------------

    @classmethod
    def from_string(cls, text: str) -> AreTomoCtfFile:
        rows = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) not in (7, 8):
                raise ValueError(f"malformed _CTF.txt line (expected 7 or 8 columns): {line!r}")
            rows.append(
                AreTomoCtfRow(
                    micrograph=int(parts[0]),
                    df_max_a=float(parts[1]),
                    df_min_a=float(parts[2]),
                    azimuth_deg=float(parts[3]),
                    phase_rad=float(parts[4]),
                    score=float(parts[5]),
                    res_a=float(parts[6]),
                    df_hand=int(parts[7]) if len(parts) == 8 else None,
                )
            )
        return cls(rows=rows)

    @classmethod
    def from_file(cls, path: str | Path) -> AreTomoCtfFile:
        return cls.from_string(Path(path).read_text())

    # -- writing ------------------------------------------------------------

    def to_string(self) -> str:
        out = [_HEADER.rstrip("\n")]
        for i, r in enumerate(self.rows):
            hand = r.df_hand if r.df_hand is not None else 1
            out.append(
                f"{i + 1:4d} {r.df_max_a:8.2f} {r.df_min_a:8.2f} {r.azimuth_deg:8.2f} "
                f"{r.phase_rad:9.4f} {r.score:8.4f} {r.res_a:8.4f} {hand:3d}"
            )
        return "\n".join(out) + "\n"

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} already exists")
        path.write_text(self.to_string())
