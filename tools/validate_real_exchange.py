#!/usr/bin/env python3
"""Validate real AreTomo3→CETS→Warp against the transferred native path.

All outputs and native reconstruction mutations are confined to --output.
The input stack and metadata are read-only. Requires external WarpTools/GPU.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import mrcfile
from lxml import etree
from cets_nonrigid.io.dose import raw_dose_from_pre_exposure
from cets_nonrigid.io.aln import raw_tilts_from_aln
from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN
from cets_nonrigid import api
from cets_nonrigid.convert import aretomo_to_warp
from cets_nonrigid.io.warp_xml import load_warp_tiltseries

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--aln", type=Path, required=True)
p.add_argument("--mdoc-dir", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--warp-tools", type=Path, required=True)
p.add_argument("--pixel-size", type=float, default=1.54)
p.add_argument("--tomo-size", nargs=3, type=int, default=(4096, 4096, 2000))
p.add_argument("--dose", type=float, default=3.87)
p.add_argument("--angpix", type=float, default=10.0)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
root = a.output.resolve()
root.mkdir(parents=True, exist_ok=a.resume)
if not a.resume:
    bundle = api.to_cets(
        "aretomo3",
        a.aln,
        mdoc_dir=a.mdoc_dir,
        pixel_size_a=a.pixel_size,
        tomo_size_px=tuple(a.tomo_size),
        dose_per_tilt=a.dose,
        cs_mm=2.7,
        amplitude_contrast=0.07,
        grid_shape=(15, 15, 5),
    )
    api.write_bundle(bundle, root / "source.cets.json")
bundle = api.read_bundle(root / "source.cets.json")
result = api.fit(bundle, "warp", movement_grid=(4, 4))
if not (root / "cets").exists():
    project = api.prepare_project(bundle, result)
    api.export_native(project, root / "cets")
name = bundle.context.parent.id
xml = root / "cets/warp_tiltseries" / f"{name}.xml"
# Normalize the native row-list contract when resuming scratch output from an
# interrupted validation run. The package writer uses this exact representation.

xml_tree = etree.parse(str(xml))
xml_tree.getroot().find("MoviePath").text = "\n".join(load_warp_tiltseries(xml).ts.tilt_movie_paths)
xml_tree.write(str(xml), xml_declaration=True, encoding="utf-8")

reference = root / "reference"
reference.mkdir(exist_ok=a.resume)
for dirname in ("tomostar", "warp_tiltseries"):
    (reference / dirname).mkdir(exist_ok=a.resume)
shutil.copy2(root / "cets/warp_tiltseries.settings", reference / "warp_tiltseries.settings")
shutil.copy2(root / "cets/tomostar" / f"{name}.tomostar", reference / "tomostar" / f"{name}.tomostar")
if not (reference / "frames").exists():
    (reference / "frames").symlink_to(root / "cets/frames", target_is_directory=True)
refxml = reference / "warp_tiltseries" / f"{name}.xml"

parsed = AreTomo3ALN.from_file(str(a.aln))
raw_rows = sorted(bundle.context.rows, key=lambda row: row.section)
raw_dose = raw_dose_from_pre_exposure(
    raw_tilts_from_aln(parsed), torch.tensor([row.accumulated_dose for row in raw_rows])
)
native = None
if not refxml.exists():
    native = aretomo_to_warp(
        a.aln,
        None,
        refxml,
        pixel_size_a=a.pixel_size,
        tomo_size_px=tuple(a.tomo_size),
        raw_dose=raw_dose,
        movement_grid=(4, 4),
        grid_shape=(15, 15, 5),
        ctf_file=a.aln.with_name(a.aln.stem + "_CTF.txt"),
        ctf_voltage_kv=300.0,
        ctf_cs_mm=2.7,
        ctf_amp_contrast=0.07,
        tilt_images=load_warp_tiltseries(xml).ts.tilt_movie_paths,
    )
first = load_warp_tiltseries(xml).model
second = load_warp_tiltseries(refxml).model
points = bundle.samples.heldout.points + torch.tensor(bundle.context.reference_center_a)
q1, v1 = first.project_volume(points)
q2, v2 = second.project_volume(points)
error = (q1.to(torch.float64) - q2.to(torch.float64))[v1 & v2].norm(dim=-1)
metrics = {
    "sample_difference_rms_a": float(error.square().mean().sqrt()),
    "sample_difference_max_a": float(error.max()),
    "cets_fit": result.metrics,
    "native_fit_rms_a_heldout": native.fit.rms_a_heldout
    if native
    else json.loads((root / "results.json").read_text())["native_fit_rms_a_heldout"],
    "source": str(a.aln.resolve()),
    "reconstruction_status": "not_evaluated",
}
(root / "results.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
if metrics["sample_difference_max_a"] > 0.01:
    raise RuntimeError(f"CETS/native sampled parity exceeded 0.01 A: {metrics}")
env = dict(
    os.environ,
    WARP_FORCE_MRC_FLOAT32="1",
    DOTNET_USE_POLLING_FILE_WATCHER="1",
    DOTNET_CLI_HOME=str(root / "dotnet-home"),
    DOTNET_CLI_TELEMETRY_OPTOUT="1",
)
for directory in (root / "cets", reference):
    command = [
        str(a.warp_tools.resolve()),
        "ts_reconstruct",
        "--settings",
        "warp_tiltseries.settings",
        "--input_data",
        f"tomostar/{name}.tomostar",
        "--angpix",
        str(a.angpix),
        "--dont_invert",
        "--device_list",
        "0",
        "--perdevice",
        "1",
    ]
    with (directory / "reconstruction.log").open("w") as log:
        run = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if run.returncode:
        raise RuntimeError(f"WarpTools failed with {run.returncode}; see {directory}/reconstruction.log")
volumes = []
for directory in (root / "cets", reference):
    files = list((directory / "warp_tiltseries/reconstruction").glob("*.mrc"))
    if len(files) != 1:
        raise RuntimeError(f"expected one reconstruction in {directory}")
    with mrcfile.open(files[0]) as f:
        volumes.append(np.array(f.data, dtype=np.float64))
x, y = volumes
if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
    raise RuntimeError("reconstruction shapes/finiteness differ")
correlation = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
relative_rms = float(np.sqrt(np.mean((x - y) ** 2)) / np.std(y))
metrics.update(
    reconstruction_status="evaluated",
    reconstruction_shape=list(x.shape),
    reconstruction_correlation=correlation,
    reconstruction_relative_rms=relative_rms,
)
(root / "results.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
if correlation < 0.999 or relative_rms > 0.01:
    raise RuntimeError("CETS/native reconstruction parity exceeded correlation/RMS gates")
print(json.dumps(metrics, indent=2, default=str))
