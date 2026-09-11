#!/usr/bin/env python3
"""Read-only comparison against the pre-port repository/environment manifests."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "PYTHONDONTWRITEBYTECODE": "1"}
results = {}
failures = []
for manifest in sorted((root / "docs/baseline").glob("*.json")):
    record = json.loads(manifest.read_text())
    if not {"path", "head", "status", "refs", "files"}.issubset(record):
        continue
    path = Path(record["path"])

    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], env=env).decode()

    errors = []
    for name, current in (
        ("head", git("rev-parse", "HEAD").strip()),
        ("status", git("status", "--short", "--untracked-files=all")),
        ("refs", git("show-ref")),
    ):
        if current != record[name]:
            errors.append(name)
    names = set(filter(None, git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")))
    expected = set(record["files"])
    if names != expected:
        errors.append({"new_files": sorted(names - expected), "missing_files": sorted(expected - names)})
    for name, saved in record["files"].items():
        file = path / name
        if file.is_symlink():
            if str(file.readlink()) != saved.get("symlink"):
                errors.append(name)
        elif not file.is_file():
            errors.append(name)
        elif hashlib.sha256(file.read_bytes()).hexdigest() != saved.get("sha256"):
            errors.append(name)
    results[manifest.stem] = {"unchanged": not errors, "files": len(expected), "differences": errors}
    failures.extend((manifest.stem, error) for error in errors)
baseline = json.loads((root / "docs/baseline/environment.json").read_text())
original_python = baseline.get("python_executable", baseline.get("python"))
if not isinstance(original_python, str) or not Path(original_python).is_file():
    original_python = "/hpc/mydata/utz.ermel/anaconda/23.1.0-3/x86_64/envs/arewarpo/bin/python"
code = "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions() if d.metadata.get('Name')}))"
current = json.loads(subprocess.check_output([original_python, "-c", code], env=env))
results["original_environment"] = {"unchanged": current == baseline["dependencies"], "packages": len(current)}
if current != baseline["dependencies"]:
    results["original_environment"]["differences"] = {
        k: [baseline["dependencies"].get(k), current.get(k)]
        for k in set(current) | set(baseline["dependencies"])
        if current.get(k) != baseline["dependencies"].get(k)
    }
    failures.append("original_environment")
print(json.dumps(results, indent=2))
raise SystemExit(bool(failures))
