#!/usr/bin/env python3
"""Build external validation tools in scratch space, without modifying source repos.

No GPL-derived source or native binary is distributed in cets-nonrigid. This
script consumes explicit external source trees and copies build inputs into a
private workspace. The resulting tools and source notices remain there.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port-source", type=Path, required=True)
parser.add_argument("--warp", type=Path, required=True)
parser.add_argument("--aretomo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cuda", type=Path, default=Path("/hpc/apps/x86_64/cuda/12.6.3_560.35.05"))
parser.add_argument("--only", choices=("cuda", "warp", "all"), default="all")
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
manifest = {}


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manifest[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()


def run(command, cwd, env=None):
    subprocess.run([str(v) for v in command], cwd=cwd, env=env, check=True)


if args.only in ("cuda", "all"):
    out = root / "mckernel"
    out.mkdir(exist_ok=True)
    copy(args.port_source / "tools/mckernel/harness.cu", out / "harness.cu")
    source = args.aretomo / "MotionCor/Correct/GCorrectPatchShift.cu"
    text = source.read_text()
    declaration = re.search(r"^static __device__ __constant__ int giSizes\[4\];.*$", text, re.M)
    kernel = re.search(r"^static __global__ void mGCorrect3D\b.*?^}", text, re.M | re.S)
    if declaration is None or kernel is None:
        raise RuntimeError("production MotionCor kernel extraction failed")
    (out / "kernel_extract.cu").write_text(
        "// Verbatim external AreTomo3 kernel; see LICENSE.AreTomo3.md\n"
        + declaration.group()
        + "\n"
        + kernel.group()
        + "\n"
    )
    manifest[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    copy(args.aretomo / "LICENSE.md", out / "LICENSE.AreTomo3.md")
    run(
        [
            args.cuda / "bin/nvcc",
            "-O2",
            "-std=c++14",
            "-gencode",
            "arch=compute_86,code=sm_86",
            "-gencode",
            "arch=compute_90,code=sm_90",
            "harness.cu",
            "-o",
            "mckernel_harness",
        ],
        out,
    )

if args.only in ("warp", "all"):
    out = root / "warpgolden"
    out.mkdir(exist_ok=True)
    external = out / "external"
    external.mkdir(exist_ok=True)
    for name in ("WarpLib", "TorchSharp"):
        destination = external / name
        if destination.exists():
            raise FileExistsError(f"{destination} already exists; choose a new scratch build")
        shutil.copytree(
            args.warp / name, destination, ignore=shutil.ignore_patterns("bin", "obj", ".git", "*.dll", "*.so", "*.pdb")
        )
        for source in (args.warp / name).rglob("*"):
            if (
                source.is_file()
                and source.suffix in {".cs", ".csproj", ".props", ".targets"}
                and "obj" not in source.parts
            ):
                manifest[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        if (args.warp / name).is_file():
            copy(args.warp / name, out / ("Warp-" + name))
    harness = out / "WarpGolden"
    harness.mkdir(exist_ok=True)
    copy(args.port_source / "tools/warpgolden/WarpGolden/Program.cs", harness / "Program.cs")
    (harness / "WarpGolden.csproj").write_text("""<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>
<OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks>
<InvariantGlobalization>true</InvariantGlobalization><Nullable>disable</Nullable></PropertyGroup>
<ItemGroup><ProjectReference Include="../external/WarpLib/WarpLib.csproj" /></ItemGroup></Project>""")
    native = out / "native"
    native.mkdir(exist_ok=True)
    copy(args.port_source / "tools/warpgolden/native/include/Functions.h", native / "include/Functions.h")
    copy(args.warp / "NativeAcceleration/src/Einspline.cpp", native / "Einspline.cpp")
    shutil.copytree(args.warp / "NativeAcceleration/src/einspline", native / "einspline")
    run(
        [
            "g++",
            "-O2",
            "-fPIC",
            "-shared",
            "-std=c++17",
            "Einspline.cpp",
            "einspline/bspline_create.cpp",
            "einspline/bspline_data.cpp",
            "-o",
            "libNativeAcceleration.so",
        ],
        native,
    )
    env = dict(
        os.environ,
        DOTNET_CLI_HOME=str(out / "dotnet-home"),
        NUGET_PACKAGES=str(out / "nuget"),
        NUGET_HTTP_CACHE_PATH=str(out / "nuget-http"),
        DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1",
        DOTNET_CLI_TELEMETRY_OPTOUT="1",
    )
    run(
        ["dotnet", "build", harness / "WarpGolden.csproj", "-c", "Release", "--artifacts-path", out / "artifacts"],
        out,
        env,
    )
(root / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(root)
