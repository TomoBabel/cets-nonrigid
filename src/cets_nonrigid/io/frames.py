"""Per-tilt frame images for Warp projects and tilt stacks from them.

Warp opens ``<tomostar dir>/<MoviePath>`` for the movie header and reads the
pixels from ``<movie dir>/average/<name>.mrc``; a symlink to the average is a
sufficient "movie". RELION reads any 2D MRC. So a frames directory holds
``<name><ext>`` (symlink) + ``average/<name>.mrc`` (float32, nz = 1).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _mrc_header(path: Path) -> dict:
    from cets_nonrigid.meta.aretomo_run import mrc_header

    return mrc_header(path)


def _symlink_to(dst: Path, src: Path) -> bool:
    """Make ``dst`` a relative symlink to ``src``. An existing symlink is
    replaced; an existing regular file (a real movie, Warp's own average) is
    left alone and ``False`` is returned."""
    if dst.resolve() == src.resolve():
        return True
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        return False
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
    return True


def frames_from_stack(
    stack: str | Path,
    frames_dir: str | Path,
    names: list[str],
    *,
    ext: str = ".mrc",
    pixel_size_a: float | None = None,
    order: list[int] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Slice ``stack`` into ``frames_dir/average/<name>.mrc`` (slice ``order[i]``
    for ``names[i]``, identity by default) and make ``frames_dir/<name><ext>``
    a symlink to the average. Existing averages of the right shape are reused;
    an existing regular movie file is never replaced."""
    import mrcfile

    stack = Path(stack)
    frames_dir = Path(frames_dir)
    avg_dir = frames_dir / "average"
    avg_dir.mkdir(parents=True, exist_ok=True)
    order = list(order) if order is not None else list(range(len(names)))
    if len(order) != len(names):
        raise ValueError("order must have one slice index per name")
    with mrcfile.mmap(str(stack), mode="r", permissive=True) as m:
        data = m.data
        nz = 1 if data.ndim == 2 else int(data.shape[0])
        voxel = float(m.voxel_size.x) if pixel_size_a is None else float(pixel_size_a)
        if max(order) >= nz:
            raise ValueError(f"{stack}: {nz} slices but frame {max(order)} requested")
        written = []
        for name, k in zip(names, order):
            avg = avg_dir / f"{name}.mrc"
            frame = frames_dir / f"{name}{ext}"
            if avg.exists() and not overwrite:
                h = _mrc_header(avg)
                if h["nz"] != 1 or (h["nx"], h["ny"]) != (int(data.shape[-1]), int(data.shape[-2])):
                    raise FileExistsError(f"{avg} exists with a different shape (use --overwrite)")
            else:
                img = np.asarray(data[k] if data.ndim == 3 else data, dtype=np.float32)
                with mrcfile.new(str(avg), overwrite=True) as out:
                    out.set_data(img)
                    out.voxel_size = voxel
            _symlink_to(frame, avg)
            written.append(frame)
    return written


def resolve_average_paths(xml_path: str | Path, movie_paths: list[str], *, frames_dir: str | Path | None = None,
                          tomostar_dir: str | Path | None = None) -> list[Path | None]:
    """Where Warp reads the pixels of each MoviePath: ``<movie dir>/average/<stem>.mrc``.
    The movie path is resolved against the tomostar directory (settings
    DataFolder, ``<root>/tomostar`` when present, else the XML's directory)
    unless ``frames_dir`` is given, which then holds the frames directly."""
    xml_path = Path(xml_path)
    if tomostar_dir is None:
        cand = xml_path.parent.parent / "tomostar"
        tomostar_dir = cand if cand.is_dir() else xml_path.parent
    out: list[Path | None] = []
    for mp in movie_paths:
        mp = str(mp).replace("\\", "/").strip()
        if not mp:
            out.append(None)
            continue
        if frames_dir is not None:
            base = Path(frames_dir)
            stem = Path(mp).stem
            avg = base / "average" / f"{stem}.mrc"
            direct = base / f"{stem}.mrc"
            out.append(avg if avg.exists() else direct)
        else:
            # normalise, never resolve: the movie may be a symlink into average/
            movie = Path(os.path.normpath(Path(tomostar_dir).absolute() / mp))
            out.append(movie.parent / "average" / f"{movie.stem}.mrc")
    return out


def stack_from_frames(files: list[Path], out: str | Path, *, pixel_size_a: float, order: list[int] | None = None,
                      overwrite: bool = False) -> Path:
    """Float32 stack with one slice per entry of ``files`` (``files[order[k]]``
    becomes slice ``k``); every image must be a single 2D layer of one shape."""
    import mrcfile

    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists")
    order = list(order) if order is not None else list(range(len(files)))
    imgs = []
    for k in order:
        p = Path(files[k])
        with mrcfile.open(str(p), permissive=True) as m:
            a = np.asarray(m.data, dtype=np.float32)
        if a.ndim == 3:
            if a.shape[0] != 1:
                raise ValueError(f"{p}: {a.shape[0]} layers, expected a single image")
            a = a[0]
        imgs.append(a)
    shapes = {a.shape for a in imgs}
    if len(shapes) != 1:
        raise ValueError(f"frames have different shapes: {sorted(shapes)}")
    with mrcfile.new(str(out), overwrite=True) as m:
        m.set_data(np.stack(imgs, axis=0))
        m.voxel_size = float(pixel_size_a)
    return out


def frames_from_images(
    images: list[Path],
    frames_dir: str | Path,
    names: list[str],
    *,
    ext: str = ".mrc",
) -> list[Path]:
    """Existing 2D images become ``frames_dir/average/<name>.mrc`` (symlink) and
    ``frames_dir/<name><ext>`` (symlink to the average); existing regular files
    are left alone."""
    frames_dir = Path(frames_dir)
    avg_dir = frames_dir / "average"
    avg_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for img, name in zip(images, names):
        img = Path(img).resolve()
        avg = avg_dir / f"{name}.mrc"
        frame = frames_dir / f"{name}{ext}"
        for dst, src in ((avg, img), (frame, avg)):
            _symlink_to(dst, src)
        out.append(frame)
    return out
