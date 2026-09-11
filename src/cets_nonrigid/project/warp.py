"""Warp project root: what ``WarpTools ts_reconstruct`` needs around a
converted tilt-series XML (route 2 of the tutorials' appendix):

  <root>/warp_tiltseries.settings         create_settings document (one per project)
  <root>/tomostar/<stem>.tomostar         movie names + angles + axis + dose (XML order)
  <root>/warp_tiltseries/<stem>.xml       the conversion (MoviePath relative to tomostar/)
  <frames dir>/<name><ext>                movie stand-ins (symlinks) + average/<name>.mrc

Nothing is executed; a hint line is returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from cryoet_alignment.io.warp.settings import WarpSettings

from cets_nonrigid.io.tomostar import tomostar_from_series
from cets_nonrigid.io.warp_settings import settings_conflicts
from cets_nonrigid.project.common import Gate, gate

SETTINGS = "warp_tiltseries.settings"
TOMOSTAR_DIR = "tomostar"
PROCESSING_DIR = "warp_tiltseries"
DEFAULT_FRAMES_DIR = "frames"


@dataclass
class WarpSeriesOutputs:
    stem: str
    xml: Path
    tomostar: Path
    settings: Path
    frames_dir: Path | None
    gates: list[Gate] = field(default_factory=list)
    hint: str = ""

    def as_dict(self) -> dict:
        return {"xml": str(self.xml), "tomostar": str(self.tomostar), "settings": str(self.settings),
                "frames_dir": str(self.frames_dir) if self.frames_dir else None}


class WarpProject:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.settings_path = self.root / SETTINGS
        self.tomostar_dir = self.root / TOMOSTAR_DIR
        self.processing_dir = self.root / PROCESSING_DIR

    def xml_path(self, stem: str) -> Path:
        return self.processing_dir / f"{stem}.xml"

    def movie_path(self, frames_dir: str | Path, name: str, ext: str) -> str:
        """MoviePath entry: the frame relative to the tomostar directory."""
        rel = os.path.relpath(Path(frames_dir).resolve(), self.tomostar_dir.resolve())
        return f"{rel}/{name}{ext}"

    def ensure_settings(self, wanted: WarpSettings) -> tuple[Path, list[str]]:
        """Write the settings once; a later series must agree on the slots
        ts_reconstruct reads (one settings per Warp project)."""
        if self.settings_path.exists():
            existing = WarpSettings.from_file(str(self.settings_path))
            conflicts = settings_conflicts(existing, wanted)
            if conflicts:
                raise ValueError(
                    f"{self.settings_path} disagrees with this series — a Warp project has one settings "
                    "file; use another project root:\n  " + "\n  ".join(conflicts)
                )
            return self.settings_path, ["settings reused"]
        self.root.mkdir(parents=True, exist_ok=True)
        wanted.to_file(str(self.settings_path))
        return self.settings_path, ["settings written"]

    def add_series(
        self,
        stem: str,
        ts,
        *,
        xml_path: Path,
        settings: WarpSettings,
        pixel_size_a: float,
        frames_dir: Path | None,
        overwrite: bool = False,
    ) -> WarpSeriesOutputs:
        tomostar = tomostar_from_series(ts)
        ts_path = self.tomostar_dir / f"{stem}.tomostar"
        if ts_path.exists() and not overwrite:
            raise FileExistsError(f"{ts_path} already exists (use --overwrite)")
        self.tomostar_dir.mkdir(parents=True, exist_ok=True)
        tomostar.to_file(str(ts_path))

        gates: list[Gate] = []
        names = tomostar.movie_names
        # normalise without following symlinks: the movie stand-in IS a symlink
        # to average/<name>.mrc, and Warp derives the average from the movie's directory
        movies = [Path(os.path.normpath(self.tomostar_dir.absolute() / n.replace("\\", "/"))) for n in names]
        missing = [n for n, m in zip(names, movies) if not m.exists()]
        gates.append(gate("frames_present", not missing, value=len(missing), expected=0,
                          note=(missing[0] if missing else "")))
        bad_avg = []
        for m in movies:
            avg = m.parent / "average" / f"{m.stem}.mrc"
            if not avg.exists():
                bad_avg.append(avg.name)
                continue
            from cets_nonrigid.meta.aretomo_run import mrc_header

            h = mrc_header(avg)
            if h["nz"] != 1 or (pixel_size_a and h["voxel"][0] > 0 and abs(h["voxel"][0] - pixel_size_a) > 1e-3 * pixel_size_a):
                bad_avg.append(avg.name)
        gates.append(gate("averages_present", not bad_avg, value=len(bad_avg), expected=0,
                          note=(bad_avg[0] if bad_avg else "average/<name>.mrc, nz=1, voxel == pix")))
        dims = settings.tomo_dims_px
        vol_a = [float(v) for v in ts.volume_dimensions_physical.tolist()]
        ok_dims = dims is not None and all(abs(d * settings.pixel_size_a - v) <= 1e-2 * max(v, 1.0) for d, v in zip(dims, vol_a))
        gates.append(gate("settings_box", ok_dims, value=dims, expected=[round(v / settings.pixel_size_a) for v in vol_a],
                          note="settings Tomo/Dimensions x PixelSize must equal VolumeDimensionsAngstrom"))
        ok_pix = settings.pixel_size_a is not None and abs(settings.pixel_size_a - pixel_size_a) <= 1e-6 * pixel_size_a
        gates.append(gate("settings_pixel_size", ok_pix, value=settings.pixel_size_a, expected=pixel_size_a))
        hint = (
            f"cd {self.root} && WARP_FORCE_MRC_FLOAT32=1 DOTNET_USE_POLLING_FILE_WATCHER=1 WarpTools ts_reconstruct "
            f"--settings {SETTINGS} --input_data {TOMOSTAR_DIR}/{stem}.tomostar --angpix <A> --dont_invert --device_list 0"
        )
        return WarpSeriesOutputs(stem=stem, xml=xml_path, tomostar=ts_path, settings=self.settings_path,
                                 frames_dir=frames_dir, gates=gates, hint=hint)
