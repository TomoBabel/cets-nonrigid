"""Warp ``.tomostar`` for a converted tilt series (thin adapter over
``cryoet_alignment.io.warp.tomostar.WarpTomostar``)."""

from __future__ import annotations

from cryoet_alignment.io.warp.tomostar import TomostarRow, WarpTomostar


def tomostar_from_series(ts, movie_names: list[str] | None = None) -> WarpTomostar:
    """Rows in XML order: ``_wrpMovieName`` = MoviePath (or ``movie_names``),
    ``_wrpAngleTilt`` = Angles, ``_wrpAxisAngle`` = AxisAngle, ``_wrpDose`` = Dose.
    Every row needs a movie name (Warp requires one; dark rows included)."""
    n = int(ts.n_tilts)
    names = list(movie_names) if movie_names is not None else list(getattr(ts, "tilt_movie_paths", None) or [])
    if len(names) != n or any(not str(x).strip() for x in names):
        raise ValueError(
            f"a tomostar needs one non-empty movie name per tilt ({n}); got {len(names)} "
            "— give --frames-dir so MoviePath entries are generated"
        )
    return WarpTomostar(rows=[
        TomostarRow(
            movie_name=str(names[i]),
            angle_tilt=float(ts.angles[i]),
            axis_angle=float(ts.tilt_axis_angles[i]),
            dose=float(ts.dose[i]),
        )
        for i in range(n)
    ])
