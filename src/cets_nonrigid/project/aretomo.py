"""AreTomo3 ``-Cmd 2`` input directory: everything is found by the stem of
``-InPrefix`` in one directory — ``<stem>.mrc`` (stack, one slice per raw
section in .aln row order), ``<stem>.aln``, ``<stem>_TLT.txt`` (read before
``.rawtlt``), ``<stem>_CTF.txt`` (``-CorrCTF 1`` only). Paths go into
256-byte buffers. Nothing is executed; a hint line is returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cryoet_alignment.io.aretomo3 import AreTomo3TLT
from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

from cets_nonrigid.io.aln_check import AlnCheck
from cets_nonrigid.project.common import Gate, failed, gate

PATH_BUFFER = 256


@dataclass
class AretomoSeriesOutputs:
    stem: str
    aln: Path
    tlt: Path | None
    ctf: Path | None
    stack: Path | None
    check: AlnCheck | None
    gates: list[Gate] = field(default_factory=list)
    hint: str = ""

    def as_dict(self) -> dict:
        return {
            "aln": str(self.aln),
            "tlt": str(self.tlt) if self.tlt else None,
            "ctf": str(self.ctf) if self.ctf else None,
            "stack": str(self.stack) if self.stack else None,
        }


def _symlink(src: Path, dst: Path) -> None:
    """``dst`` becomes a relative symlink to ``src`` (an existing entry is replaced)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def _ctf_line_gates(ctf_path: Path, n_raw: int) -> list[Gate]:
    """AreTomo3's CTF loader skips '#' lines, drops each data line's first
    character and counts every other line, blank ones included."""
    lines = ctf_path.read_text().splitlines()
    data = [ln for ln in lines if not ln.lstrip().startswith("#")]
    blank = sum(1 for ln in data if not ln.strip())
    rows = len(data) - blank
    leading = all(ln.startswith(" ") for ln in data if ln.strip())
    return [
        gate("ctf_rows", rows == n_raw, value=rows, expected=n_raw),
        gate("ctf_no_blank_lines", blank == 0, value=blank, expected=0),
        gate("ctf_leading_space", leading, value=leading, expected=True,
             note="AreTomo3 discards the first character of every data line"),
    ]


def aretomo3_hint(
    root: Path, stem: str, *, pixel_size_a=None, voltage_kv=None, cs_mm=None,
    amplitude_contrast=None, vol_z_px=None, has_ctf: bool = False,
) -> str:
    """The ``-Cmd 2`` line for the written directory. ``-AtBin`` is the user's
    choice (a placeholder is printed); ``-CorrCTF 1`` is suggested only when a
    ``_CTF.txt`` was written."""
    def f(v, spec):
        return f"{v:{spec}}" if v is not None else "<?>"

    return (
        f"AreTomo3 -Cmd 2 -Serial 0 -InPrefix {root / stem}.mrc -OutDir {root / 'out'}/ -Gpu 0 "
        f"-PixSize {f(pixel_size_a, 'g')} -kV {f(voltage_kv, 'g')} -Cs {f(cs_mm, 'g')} "
        f"-AmpContrast {f(amplitude_contrast, 'g')} -VolZ {f(vol_z_px, 'd') if vol_z_px is not None else '<?>'} "
        f"-AtBin <bin> -FlipVol 1 -Wbp 1 -CorrCTF {1 if has_ctf else 0} -SplitSum 0"
    )


class AretomoDir:
    """One AreTomo3 input directory; conversions write ``<stem>.aln`` (+
    ``_CTF.txt``) themselves and then ``finalize_series`` adds the companions
    and runs the directory gates."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def series_paths(self, stem: str) -> dict[str, Path]:
        return {
            "aln": self.root / f"{stem}.aln",
            "tlt": self.root / f"{stem}_TLT.txt",
            "ctf": self.root / f"{stem}_CTF.txt",
        }

    def prepare(self, stem: str, *, overwrite: bool) -> None:
        """Make room for a series (or refuse when it exists)."""
        self.root.mkdir(parents=True, exist_ok=True)
        existing = [p for p in self.root.glob(f"{stem}*")
                    if p.name in {f"{stem}.aln", f"{stem}_TLT.txt", f"{stem}_CTF.txt", f"{stem}.rawtlt"}
                    or (p.stem == stem and p.suffix in (".mrc", ".st", ".mrcs"))]
        if existing and not overwrite:
            raise FileExistsError(
                f"series {stem!r} already exists in {self.root} ({', '.join(p.name for p in existing)}); "
                "use --overwrite to replace it"
            )
        for p in existing:
            p.unlink()

    def finalize_series(
        self,
        stem: str,
        aln: AreTomo3ALN,
        *,
        aln_path: Path,
        aln_check: AlnCheck | None,
        tlt: AreTomo3TLT | None,
        ctf_path: Path | None,
        stack: Path | None,
        hint_kwargs: dict | None = None,
    ) -> AretomoSeriesOutputs:
        n_raw = int(aln.RawSize[2])
        gates: list[Gate] = list(aln_check.gates) if aln_check is not None else []

        out_tlt = None
        if tlt is not None:
            if tlt.n_rows != n_raw:
                raise ValueError(f"_TLT.txt has {tlt.n_rows} rows for {n_raw} raw sections")
            tlt_path = aln_path.with_name(f"{stem}_TLT.txt")
            tlt.to_file(str(tlt_path))
            out_tlt = tlt_path
            stale = aln_path.with_name(f"{stem}.rawtlt")
            if stale.exists():
                other = [float(x) for x in stale.read_text().split()]
                same = len(other) == n_raw and all(abs(a - b) <= 0.01 for a, b in zip(other, tlt.tilts))
                gates.append(gate("no_conflicting_rawtlt", same, value=stale.name,
                                  expected="absent or identical to _TLT.txt",
                                  note="AreTomo3 reads _TLT.txt first; a differing .rawtlt is confusing"))

        if ctf_path is not None and Path(ctf_path).exists():
            gates.extend(_ctf_line_gates(Path(ctf_path), n_raw))

        out_stack = None
        if stack is not None:
            stack = Path(stack)
            stack_path = aln_path.with_name(f"{stem}{stack.suffix if stack.suffix in ('.mrc', '.st', '.mrcs') else '.mrc'}")
            if stack.resolve() != stack_path.resolve():
                _symlink(stack, stack_path)
            out_stack = stack_path
            from cets_nonrigid.meta.aretomo_run import mrc_header

            h = mrc_header(stack_path)
            gates.append(gate("stack_sections", h["nz"] == n_raw, value=h["nz"], expected=n_raw))
            gates.append(gate("stack_dims", (h["nx"], h["ny"]) == (int(aln.RawSize[0]), int(aln.RawSize[1])),
                              value=(h["nx"], h["ny"]), expected=(int(aln.RawSize[0]), int(aln.RawSize[1]))))
            gates.append(gate("stack_mode", h["mode"] in (0, 1, 2, 6), value=h["mode"], expected="0/1/2/6"))
            pix = (hint_kwargs or {}).get("pixel_size_a")
            if pix is not None and h["voxel"][0] > 0:
                ok = abs(h["voxel"][0] - pix) <= 1e-3 * pix
                gates.append(gate("stack_voxel", ok, value=round(h["voxel"][0], 4), expected=pix,
                                  note="-PixSize overrides the header" if not ok else ""))

        longest = len(str(aln_path.resolve().with_suffix(""))) + len("_CTF.txt")
        gates.append(gate("path_length", longest < PATH_BUFFER, value=longest, expected=f"< {PATH_BUFFER}",
                          note="AreTomo3 builds input paths in 256-byte buffers"))

        hk = dict(hint_kwargs or {})
        if hk.get("vol_z_px") is None:
            hk["vol_z_px"] = int(aln.Thickness) if aln.Thickness else None
        hk["has_ctf"] = ctf_path is not None and Path(ctf_path).exists()
        hint = aretomo3_hint(aln_path.parent, stem, **hk)
        return AretomoSeriesOutputs(
            stem=stem, aln=aln_path, tlt=out_tlt, ctf=Path(ctf_path) if ctf_path else None,
            stack=out_stack, check=aln_check, gates=gates, hint=hint,
        )


def any_failed(outputs: AretomoSeriesOutputs) -> list[Gate]:
    return failed(outputs.gates)
