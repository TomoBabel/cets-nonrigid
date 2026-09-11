"""The CLI's small spec grammars, kept as pure functions.

* ``SOURCE`` tokens: local files, directories (globbed by kind) or portal runs
  ``portal:<dataset-id>[/<run-name>][@alignment=ID,voxel=A]``.
* ``--particles``: a picks file, a directory of per-series picks,
  ``copick:<config.json>#<object>:<user>/<session>`` or
  ``portal:<object name>`` / ``portal:<annotation-id>``.
* ``--particles-voxel``: a voxel size (A) or a directory of tomograms.
* frame extension inference for a frames directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import click

SOURCE_PATTERNS = {"aln": "*.aln", "xml": "*.xml"}
FRAME_EXTS = (".mrc", ".mrcs", ".tif", ".tiff", ".eer")
PICK_SUFFIXES = (".star", ".ndjson", ".txt", ".csv", ".xyz")

PORTAL_SOURCE_FORMS = "portal:<dataset-id>[/<run-name>][@alignment=ID,voxel=A]"
PARTICLE_FORMS = (
    "a picks file (.star | .ndjson | 'x y z' text), a directory of <stem>.star|.ndjson|.txt, "
    "copick:<config.json>#<object>:<user>/<session>, or portal:<object name> | portal:<annotation-id>"
)


# ---------------------------------------------------------------------------
# SOURCE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortalSource:
    dataset_id: int
    run_name: str | None = None
    alignment_id: int | None = None
    voxel_spacing: float | None = None

    def __str__(self) -> str:
        s = f"portal:{self.dataset_id}" + (f"/{self.run_name}" if self.run_name else "")
        opts = [f"alignment={self.alignment_id}" if self.alignment_id is not None else None,
                f"voxel={self.voxel_spacing:g}" if self.voxel_spacing is not None else None]
        opts = [o for o in opts if o]
        return s + (f"@{','.join(opts)}" if opts else "")


_PORTAL_RE = re.compile(r"^portal:(?P<dataset>\d+)(?:/(?P<run>[^@]+))?(?:@(?P<opts>.*))?$")


def parse_portal_source(token: str) -> PortalSource:
    m = _PORTAL_RE.match(token.strip())
    if not m:
        raise click.UsageError(f"{token!r}: portal sources are {PORTAL_SOURCE_FORMS}")
    alignment = voxel = None
    for part in (m.group("opts") or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        try:
            if key == "alignment" and val:
                alignment = int(val)
            elif key == "voxel" and val:
                voxel = float(val)
            else:
                raise ValueError
        except ValueError:
            raise click.UsageError(f"{token!r}: unknown portal option {part!r} (alignment=ID, voxel=A)") from None
    run = m.group("run")
    return PortalSource(int(m.group("dataset")), run.strip() if run else None, alignment, voxel)


def expand_sources(tokens, kind: str, *, portal: bool = False) -> list:
    """Files are taken as given, directories are globbed (``*.aln`` / ``*.xml``),
    ``portal:`` tokens become :class:`PortalSource` (AreTomo3 sources only)."""
    pat = SOURCE_PATTERNS[kind]
    found: list = []
    for tok in tokens:
        tok = str(tok)
        if tok.startswith("portal:"):
            if not portal:
                raise click.UsageError(
                    f"{tok!r}: portal runs are AreTomo3 sources (a2w, a2r); this command reads {pat} files"
                )
            found.append(parse_portal_source(tok))
            continue
        p = Path(tok)
        if p.is_dir():
            hits = sorted(q for q in p.glob(pat) if q.is_file())
            if not hits:
                raise click.UsageError(f"no files match {pat!r} in {p}")
            found.extend(hits)
        elif p.is_file():
            found.append(p)
        else:
            raise click.UsageError(f"{tok}: not a file, a directory or a {PORTAL_SOURCE_FORMS} spec")
    seen: set = set()
    uniq: list = []
    for item in found:
        key = item.resolve() if isinstance(item, Path) else item
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    if not uniq:
        raise click.UsageError("no sources given")
    return uniq


# ---------------------------------------------------------------------------
# --particles / --particles-voxel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleSpec:
    kind: str  # file | dir | copick | portal
    path: Path | None = None
    config: Path | None = None
    uri: str | None = None
    object_name: str | None = None
    annotation_id: int | None = None

    def __str__(self) -> str:
        if self.kind == "copick":
            return f"copick:{self.config}#{self.uri}"
        if self.kind == "portal":
            return f"portal:{self.annotation_id if self.annotation_id is not None else self.object_name}"
        return str(self.path)


def parse_particles_spec(text: str) -> ParticleSpec:
    text = str(text).strip()
    if text.startswith("copick:"):
        config, sep, uri = text[len("copick:"):].partition("#")
        if not sep or not config or not uri:
            raise click.UsageError(f"{text!r}: copick picks are copick:<config.json>#<object>:<user>/<session>")
        cfg = Path(config)
        if not cfg.is_file():
            raise click.UsageError(f"{text!r}: copick config {cfg} not found")
        return ParticleSpec("copick", config=cfg, uri=uri.strip())
    if text.startswith("portal:"):
        body = text[len("portal:"):].strip()
        if not body:
            raise click.UsageError(f"{text!r}: portal picks are portal:<object name> or portal:<annotation-id>")
        if body.isdigit():
            return ParticleSpec("portal", annotation_id=int(body))
        return ParticleSpec("portal", object_name=body)
    p = Path(text)
    if p.is_dir():
        return ParticleSpec("dir", path=p)
    if p.is_file():
        return ParticleSpec("file", path=p)
    raise click.UsageError(f"{text!r}: --particles takes {PARTICLE_FORMS}")


def parse_particles_voxel(text: str):
    """A voxel size in A (float) or a directory of tomograms (Path)."""
    text = str(text).strip()
    try:
        v = float(text)
    except ValueError:
        v = None
    if v is not None:
        if v <= 0:
            raise click.UsageError(f"--particles-voxel {text}: the voxel size must be positive")
        return v
    p = Path(text)
    if p.is_dir():
        return p
    raise click.UsageError(f"--particles-voxel {text!r}: give a voxel size in A or a directory of tomograms <stem>*.mrc")


def particles_spec_callback(ctx, param, value):
    return parse_particles_spec(value) if value else None


def particles_voxel_callback(ctx, param, value):
    return parse_particles_voxel(value) if value else None


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def infer_frames_ext(frames_dir, names) -> str | None:
    """The extension under which frames ``<name><ext>`` already exist in
    ``frames_dir`` (first of ``.mrc .mrcs .tif .tiff .eer`` with any hit);
    ``None`` when nothing is there yet."""
    if frames_dir is None or not names:
        return None
    d = Path(frames_dir)
    if not d.is_dir():
        return None
    for ext in FRAME_EXTS:
        if any((d / f"{n}{ext}").exists() for n in names):
            return ext
    return None


def find_by_stem(directory, stem: str, suffixes) -> Path | None:
    """``<directory>/<stem><suffix>`` (first suffix that exists), else the
    first ``<stem>*`` file with one of the suffixes."""
    if directory is None:
        return None
    directory = Path(directory)
    for suf in suffixes:
        cand = directory / f"{stem}{suf}"
        if cand.exists():
            return cand
    hits = sorted(directory.glob(f"{stem}*"))
    hits = [h for h in hits if h.suffix.lower() in suffixes]
    return hits[0] if hits else None
