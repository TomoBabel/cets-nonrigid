#!/usr/bin/env python3
"""Install both development repositories into a NEW isolated environment.

Run with Python 3.13 on Linux x86_64. No existing environment is modified.
Custom CUDA wheels are installed before the pinned native backend. --dry-run
prints the exact commands without creating anything.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", type=Path, required=True)
parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parents[2] / "cets-data-models-dev")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "environment-lock.json").read_text())
if args.environment.exists():
    raise SystemExit("Choose a new environment path; existing environments are never modified.")
if not (args.schema / "pyproject.toml").is_file():
    raise SystemExit("The independent development schema checkout is required.")
schema_commit = subprocess.check_output(["git", "-C", str(args.schema), "rev-parse", "HEAD"], text=True).strip()
if schema_commit != lock["repositories"]["cets-data-models-dev"]:
    raise SystemExit("Schema checkout does not match environment-lock.json; reconcile and revalidate first.")
python = args.environment.resolve() / "bin/python"
pip = [str(python), "-m", "pip", "install"]
constraints = ["-c", str(root / "constraints-linux-cu129.txt")]
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
# Git dependencies are installed explicitly above the numerical package, including
# the exact-commit editable schema. Resolve the remaining requirements first so a
# final --no-deps editable install cannot replace that schema with the branch URL.
ordinary_dependencies = [requirement for requirement in project["dependencies"] if " @ " not in requirement]
development_dependencies = project["optional-dependencies"]["test"] + project["optional-dependencies"]["dev"]
commands = [
    [sys.executable, "-m", "venv", str(args.environment.resolve())],
    pip + constraints + ["setuptools", "wheel", "build"],
    pip + ["torch==2.8.0+cu129", "--index-url", lock["torch_index"]],
    pip + ["torch-projectors==0.13.0+cu129", "--no-deps", "--index-url", lock["projectors_index"]],
    pip + constraints + ["torch-subpixel-crop", "pillow", "lxml", "starfile>=0.5,<0.6"],
    pip + ["--no-deps", "warpylib @ git+https://github.com/uermel/warpylib.git@" + lock["repositories"]["warpylib"]],
    pip
    + constraints
    + [
        "cryoet-alignment @ git+https://github.com/uermel/cryoet-alignment.git@"
        + lock["repositories"]["cryoet-alignment"]
    ],
    pip + constraints + ["-e", str(args.schema.resolve())],
    pip
    + constraints
    + ordinary_dependencies
    + development_dependencies
    + ["linkml==1.11.1", "linkml-runtime==1.11.1", "deepdiff"],
    pip + ["--no-deps", "-e", str(root)],
    [str(python), "-m", "pip", "check"],
    [str(python), str(root / "tools/check_environment.py")],
]
for command in commands:
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
