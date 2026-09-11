"""RELION 5 / py2rely project root.

Layout (RELION and py2rely resolve star-internal paths from the project root):

  <root>/tomograms.star             data_global, one row per series
  <root>/tilt_series/<name>.star    data_<name>
  <root>/particles.star             general + optics (one group per distinct
                                    kV/Cs/amp/pixel tuple) + particles
  <root>/motion.star                general(rlnParticleNumber) + one block per particle
  <root>/optimisation_set.star      root-relative paths; particles/trajectories
                                    entries only when the project has them
  [<root>/tilt_series/tiltseries_placeholder.mrcs]  sparse (py2rely/ZPT streaming)

Series are appended (or replaced with ``overwrite``); ``flush`` rewrites the
project files. ``# version 50001`` on every file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import starfile
import torch

from cets_nonrigid.io.relion_star import (
    VERSION_COMMENT,
    RelionTomogramData,
    motion_blocks,
    particles_frames,
    read_star_tables,
    tilt_series_frame,
    tomogram_global_row,
    write_optimisation_set,
    write_star_tables,
)

TOMOGRAMS = "tomograms.star"
PARTICLES = "particles.star"
MOTION = "motion.star"
OPTSET = "optimisation_set.star"
TILT_DIR = "tilt_series"
PLACEHOLDER = "tiltseries_placeholder.mrcs"
URI_LABEL = "tomoTiltSeriesURI"


def random_subset_hash(name: str) -> int:
    """Half-set from the particle name: stable across re-runs, arms and subsets."""
    return 1 + int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 2


@dataclass
class RelionSeriesEntry:
    name: str
    tilt_star: Path
    n_tilts: int
    n_particles: int
    optics_group: int
    micrograph_names: list
    tilt_star_bytes: bytes
    particles: pd.DataFrame | None = None
    motion: dict = field(default_factory=dict)

    def outputs(self, root: Path) -> dict:
        return {
            "tomograms_star": str(root / TOMOGRAMS),
            "tilt_series_star": str(self.tilt_star),
            "particles_star": str(root / PARTICLES) if self.n_particles else None,
            "motion_star": str(root / MOTION) if self.motion else None,
            "optimisation_set": str(root / OPTSET),
        }


class RelionProject:
    """Append-only view of a RELION project root; call ``flush`` to write."""

    def __init__(self, root: str | Path, *, random_subset: str = "hash"):
        self.root = Path(root)
        if random_subset not in ("hash", "alternate"):
            raise ValueError("random_subset must be 'hash' or 'alternate'")
        self.random_subset = random_subset
        self.global_rows: list[dict] = []
        self.optics: list[dict] = []
        self.particles: pd.DataFrame | None = None
        self.motion: dict[str, pd.DataFrame] = {}
        self.general: dict = {"rlnTomoSubTomosAre2DStacks": 0}
        self.entries: dict[str, RelionSeriesEntry] = {}
        self.dirty = False
        self._load()

    # --- existing project ------------------------------------------------------

    def _load(self) -> None:
        tpath = self.root / TOMOGRAMS
        if tpath.exists():
            g = read_star_tables(tpath).get("global")
            if isinstance(g, dict):
                g = pd.DataFrame([g])
            if g is not None:
                self.global_rows = [dict(r) for _, r in g.iterrows()]
        ppath = self.root / PARTICLES
        if ppath.exists():
            blocks = read_star_tables(ppath)
            if "general" in blocks:
                gen = blocks["general"]
                self.general = dict(gen) if isinstance(gen, dict) else dict(gen.iloc[0])
            optics = blocks.get("optics")
            if isinstance(optics, dict):
                optics = pd.DataFrame([optics])
            if optics is not None:
                self.optics = [dict(r) for _, r in optics.iterrows()]
            parts = blocks.get("particles")
            if isinstance(parts, dict):
                parts = pd.DataFrame([parts])
            self.particles = parts
        mpath = self.root / MOTION
        if mpath.exists():
            blocks = read_star_tables(mpath)
            self.motion = {k: v for k, v in blocks.items() if k != "general"}

    @property
    def series_names(self) -> list[str]:
        return [str(r["rlnTomoName"]) for r in self.global_rows]

    def has_series(self, name: str) -> bool:
        return name in self.series_names

    @property
    def n_particles(self) -> int:
        return 0 if self.particles is None else len(self.particles)

    # --- edits -----------------------------------------------------------------

    def remove_series(self, name: str) -> None:
        self.global_rows = [r for r in self.global_rows if str(r["rlnTomoName"]) != name]
        for p in (self.root / TILT_DIR / f"{name}.star", self.root / TILT_DIR / f"{name}.mrcs"):
            if p.exists() or p.is_symlink():
                p.unlink()
        if self.particles is not None and "rlnTomoName" in self.particles.columns:
            gone = self.particles[self.particles["rlnTomoName"].astype(str) == name]
            for pname in gone["rlnTomoParticleName"].astype(str):
                self.motion.pop(pname, None)
            self.particles = self.particles[self.particles["rlnTomoName"].astype(str) != name].reset_index(drop=True)
            if len(self.particles) == 0:
                self.particles = None
        self.entries.pop(name, None)
        self.dirty = True

    def _optics_group(self, tomo: RelionTomogramData) -> int:
        key = (round(tomo.voltage_kv, 3), round(tomo.cs_mm, 4), round(tomo.amplitude_contrast, 4),
               round(tomo.pixel_size_a, 5))
        for row in self.optics:
            rk = (round(float(row["rlnVoltage"]), 3), round(float(row["rlnSphericalAberration"]), 4),
                  round(float(row["rlnAmplitudeContrast"]), 4), round(float(row["rlnTomoTiltSeriesPixelSize"]), 5))
            if rk == key:
                return int(row["rlnOpticsGroup"])
        k = len(self.optics) + 1
        self.optics.append({
            "rlnOpticsGroup": k, "rlnOpticsGroupName": f"opticsGroup{k}", "rlnVoltage": tomo.voltage_kv,
            "rlnSphericalAberration": tomo.cs_mm, "rlnAmplitudeContrast": tomo.amplitude_contrast,
            "rlnTomoTiltSeriesPixelSize": tomo.pixel_size_a,
        })
        return k

    def add_series(
        self,
        tomo: RelionTomogramData,
        *,
        particle_names: list | None = None,
        centered_coords_a: torch.Tensor | None = None,
        motion_a: torch.Tensor | None = None,
        tilt_series_uri: str | None = None,
        point_attributes: dict | None = None,
        overwrite: bool = False,
    ) -> RelionSeriesEntry:
        name = tomo.name
        if self.has_series(name):
            if not overwrite:
                raise FileExistsError(
                    f"series {name!r} already exists in {self.root} (use --overwrite to replace it)"
                )
            self.remove_series(name)
        if particle_names and self.particles is not None:
            dup = set(particle_names) & set(self.particles["rlnTomoParticleName"].astype(str))
            if dup:
                raise ValueError(f"duplicate rlnTomoParticleName across the project: {sorted(dup)[:3]}...")

        k = self._optics_group(tomo)
        tomo.optics_group_name = f"opticsGroup{k}"
        ts_ref = f"{TILT_DIR}/{name}.star"
        ts_path = self.root / ts_ref
        frame = tilt_series_frame(tomo)
        row = tomogram_global_row(tomo, ts_ref)
        if tilt_series_uri is not None:
            frame[URI_LABEL] = tilt_series_uri
            row[URI_LABEL] = tilt_series_uri
        write_star_tables(ts_path, {name: frame}, overwrite=True)
        self.global_rows.append(row)

        entry = RelionSeriesEntry(
            name=name, tilt_star=ts_path, n_tilts=tomo.n_tilts, n_particles=0, optics_group=k,
            micrograph_names=list(tomo.micrograph_names or []), tilt_star_bytes=ts_path.read_bytes(),
        )
        if particle_names:
            if centered_coords_a is None:
                raise ValueError("particle_names given without coordinates")
            subset = [random_subset_hash(n) for n in particle_names] if self.random_subset == "hash" else None
            _general, _optics, parts = particles_frames(
                tomo_name=name, particle_names=particle_names, centered_coords_a=centered_coords_a,
                voltage_kv=tomo.voltage_kv, cs_mm=tomo.cs_mm, amplitude_contrast=tomo.amplitude_contrast,
                pixel_size_a=tomo.pixel_size_a, optics_group=k, optics_group_name=f"opticsGroup{k}",
                random_subset=subset,
            )
            for key, values in (point_attributes or {}).items():
                label = {"half_set": "rlnRandomSubset", "class_number": "rlnClassNumber"}.get(key)
                if label is not None:
                    if len(values) != len(parts):
                        raise ValueError(f"particle attribute {key} has wrong length")
                    if key == "half_set" and not set(values) <= {1, 2}:
                        raise ValueError("RELION half sets must be 1 or 2")
                    parts[label] = values
            self.particles = parts if self.particles is None else pd.concat([self.particles, parts], ignore_index=True)
            if self.random_subset == "alternate":
                self.particles["rlnRandomSubset"] = (pd.RangeIndex(len(self.particles)) % 2 + 1).to_numpy()
            entry.particles = parts
            entry.n_particles = len(parts)
            if motion_a is not None:
                blocks = motion_blocks(particle_names, motion_a)
                self.motion.update(blocks)
                entry.motion = blocks
        self.entries[name] = entry
        self.dirty = True
        return entry

    # --- stack references ---------------------------------------------------------

    def link_stack(self, name: str, stack: str | Path, *, link: str = "symlink") -> str:
        """Make ``tilt_series/<name>.mrcs`` refer to the tilt stack and return
        that root-relative reference. RELION reads ``N@file`` only from
        ``.mrcs`` files (image.h: ``.mrc`` is reserved for 3D maps), so the
        stack is linked under that extension; nothing is copied unless asked."""
        import os
        import shutil

        stack = Path(stack)
        dst = self.root / TILT_DIR / f"{name}.mrcs"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link == "symlink":
            dst.symlink_to(os.path.relpath(stack.resolve(), dst.parent.resolve()))
        elif link == "hardlink":
            os.link(stack, dst)
        elif link == "copy":
            shutil.copy2(stack, dst)
        else:
            raise ValueError(f"unknown link mode {link!r}")
        return f"{TILT_DIR}/{name}.mrcs"

    # --- placeholder stack (py2rely / ZPT streaming) -----------------------------

    def placeholder_stack(self, *, nx: int, ny: int, nz: int, pixel_size_a: float) -> Path:
        """Sparse float32 header-only stack behind ``N@`` names (never read for
        pixels; RELION only opens the header)."""
        import mrcfile

        path = self.root / TILT_DIR / PLACEHOLDER
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with mrcfile.open(str(path), header_only=True, permissive=True) as m:
                have = (int(m.header.nx), int(m.header.ny), int(m.header.nz))
            if have[0] == nx and have[1] == ny and have[2] >= nz:
                return path
            path.unlink()
        with mrcfile.new_mmap(str(path), shape=(nz, ny, nx), mrc_mode=2, overwrite=True) as m:
            m.voxel_size = (pixel_size_a, pixel_size_a, 1.0)
            m.update_header_from_data()
        return path

    @staticmethod
    def placeholder_ref() -> str:
        return f"{TILT_DIR}/{PLACEHOLDER}"

    # --- write ------------------------------------------------------------------

    def flush(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.global_rows:
            raise ValueError("project has no series")
        names = [str(r["rlnTomoName"]) for r in self.global_rows]
        if len(set(names)) != len(names):
            raise ValueError("duplicate rlnTomoName in the project")
        write_star_tables(self.root / TOMOGRAMS, {"global": pd.DataFrame(self.global_rows)}, overwrite=True)
        outputs = {"tomograms_star": str(self.root / TOMOGRAMS), "particles_star": None,
                   "motion_star": None, "optimisation_set": str(self.root / OPTSET)}
        if self.particles is not None and len(self.particles):
            write_star_tables(
                self.root / PARTICLES,
                {"general": pd.DataFrame([self.general]), "optics": pd.DataFrame(self.optics),
                 "particles": self.particles},
                overwrite=True,
            )
            outputs["particles_star"] = str(self.root / PARTICLES)
            if self.motion:
                order = [n for n in self.particles["rlnTomoParticleName"].astype(str) if n in self.motion]
                missing = [n for n in self.particles["rlnTomoParticleName"].astype(str) if n not in self.motion]
                if missing:
                    raise ValueError(f"motion.star would miss {len(missing)} particles (e.g. {missing[0]!r})")
                blocks = {"general": {"rlnParticleNumber": len(order)}, **{n: self.motion[n] for n in order}}
                write_star_tables(self.root / MOTION, blocks, overwrite=True)
                outputs["motion_star"] = str(self.root / MOTION)
            elif (self.root / MOTION).exists():
                (self.root / MOTION).unlink()
        else:
            for f in (PARTICLES, MOTION):
                if (self.root / f).exists():
                    (self.root / f).unlink()
        write_optimisation_set(
            self.root / OPTSET,
            particles=PARTICLES if outputs["particles_star"] else None,
            tomograms=TOMOGRAMS,
            trajectories=MOTION if outputs["motion_star"] else None,
            overwrite=True,
        )
        self.dirty = False
        return outputs

    @staticmethod
    def star_bytes(blocks: dict) -> bytes:
        return (VERSION_COMMENT + "\n" + starfile.to_string(blocks)).encode()
