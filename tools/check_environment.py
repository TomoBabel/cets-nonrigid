#!/usr/bin/env python3
"""Check the supported numerical ABI and report the active development packages."""

import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch
import warpylib
from cets_data_model.models import models as m
import cets_nonrigid
import cryoet_alignment

if torch.__version__ != "2.8.0+cu129":
    raise RuntimeError(f"validated environment requires torch 2.8.0+cu129; found {torch.__version__}")
if metadata.version("torch-projectors") != "0.13.0+cu129":
    raise RuntimeError("validated environment requires torch-projectors 0.13.0+cu129")
if warpylib.TiltSeries is None:
    raise RuntimeError("warpylib TiltSeries failed to import; check the torch-projectors ABI")
if "non_rigid_alignment" not in m.Alignment.model_fields:
    raise RuntimeError("install the development CETS schema with non-rigid alignment support")
if importlib.util.find_spec("arewarpion") is not None:
    raise RuntimeError("independence check requires an environment without arewarpion")
for name in ("cets_warpm", "cets_aretomo3", "cets_relion", "cets_imod"):
    if importlib.util.find_spec(name) is not None:
        raise RuntimeError(f"independence check found downstream converter {name}")


def revision(module):
    path = Path(module.__file__).resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            return {
                "path": str(parent),
                "commit": subprocess.check_output(["git", "-C", str(parent), "rev-parse", "HEAD"], text=True).strip(),
            }
    return {"path": str(path), "commit": None}


print(
    json.dumps(
        {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_projectors": metadata.version("torch-projectors"),
            "cuda_available": torch.cuda.is_available(),
            "schema": revision(m),
            "numerical": revision(cets_nonrigid),
            "codec": revision(cryoet_alignment),
            "warpylib": revision(warpylib),
        },
        indent=2,
    )
)
