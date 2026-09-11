"""Strict Warp movie-XML loading.

Mirrors the tilt-series strict adapter (``io/warp_xml.py``): validate the raw
XML first, then load through warpylib and cross-check. Warp movie XML stores
NO image dimensions, frame count, or FractionFrames — those are runtime inputs
supplied externally wherever trajectories are evaluated
(``models/warp_movie.py``), so this loader validates only what the file itself
owns: structure, motion grids, and CTF scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree
from warpylib.movie import Movie
from warpylib.movie.io import load_meta

_REQUIRED_CHILDREN = ("GridMovementX", "GridMovementY")


@dataclass
class WarpMovieXml:
    movie: Movie
    xml_bytes: bytes


def load_warp_movie_strict(xml_path: str | Path) -> WarpMovieXml:
    """Load and validate a Warp movie XML (alignment record only — pairing it
    with the raw movie and runtime metadata is the caller's responsibility)."""
    xml_path = Path(xml_path)
    xml_bytes = xml_path.read_bytes()

    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"{xml_path}: not valid XML: {e}") from e

    if root.tag != "Movie":
        raise ValueError(f"{xml_path}: root element is <{root.tag}>, expected <Movie>")

    for child in _REQUIRED_CHILDREN:
        if root.find(child) is None:
            raise ValueError(f"{xml_path}: missing <{child}> element")

    movie = Movie()
    load_meta(movie, str(xml_path))

    for name in ("grid_movement_x", "grid_movement_y"):
        grid = getattr(movie, name, None)
        if grid is None or grid.values is None:
            raise ValueError(f"{xml_path}: warpylib failed to load {name} silently")
    if not (movie.ctf.pixel_size > 0):
        raise ValueError(f"{xml_path}: non-positive CTF pixel size {movie.ctf.pixel_size}")

    return WarpMovieXml(movie=movie, xml_bytes=xml_bytes)
