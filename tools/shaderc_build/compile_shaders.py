#!/usr/bin/env python3
"""
compile_shaders.py
-------------------

Lightweight wrapper around glslc that turns a list of GLSL compute shaders
into SPIR-V binaries, writes an optional manifest, and mirrors the outputs
into an assets directory. The script is intentionally small so it can run
cross-platform without extra dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ShaderOutput:
    source: Path
    output: Path
    asset: Path
    depfile: Optional[Path]


def discover_glslc(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env_path = os.environ.get("GLSLC")
    if env_path:
        return env_path
    glslc = shutil.which("glslc")
    if glslc:
        return glslc
    raise RuntimeError(
        "glslc not found. Install the Vulkan SDK or point GLSLC to its executable."
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_for_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def compile_one(
    glslc: str,
    src: Path,
    out: ShaderOutput,
    includes: Iterable[Path],
    defines: Iterable[str],
    debug: bool,
) -> float:
    ensure_dir(out.output.parent)
    if out.asset:
        ensure_dir(out.asset.parent)
    if out.depfile:
        ensure_dir(out.depfile.parent)

    cmd: List[str] = [
        glslc,
        "-c",
        "--target-env=vulkan1.3",
        "-O",
        "-o",
        str(out.output),
        str(src),
    ]
    if debug:
        cmd.append("-g")
    for inc in includes:
        cmd.extend(("-I", str(inc)))
    for define in defines:
        cmd.append(f"-D{define}")
    if out.depfile:
        cmd.extend(("-MD", "-MF", str(out.depfile)))

    start = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"glslc failed for {src}")
    if out.asset:
        shutil.copy2(out.output, out.asset)
    return elapsed_ms


def build_outputs(
    sources: Iterable[Path],
    source_dir: Path,
    output_dir: Path,
    asset_dir: Optional[Path],
    dep_dir: Optional[Path],
) -> List[ShaderOutput]:
    outputs: List[ShaderOutput] = []
    for src in sources:
        rel = src.relative_to(source_dir)
        out_path = output_dir / (str(rel) + ".spv")
        asset_path = asset_dir / (str(rel) + ".spv") if asset_dir else Path()
        depfile = None
        if dep_dir:
            depfile = dep_dir / (str(rel) + ".d")
        outputs.append(
            ShaderOutput(source=src, output=out_path, asset=asset_path, depfile=depfile)
        )
    return outputs


def write_manifest(manifest: Path, entries: List[dict]) -> None:
    ensure_dir(manifest.parent)
    payload = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shaders": entries,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile GLSL compute shaders with glslc.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest to write.")
    parser.add_argument("--glslc", type=str, help="Path to glslc executable.")
    parser.add_argument(
        "--define",
        action="append",
        default=[],
        help="Additional -D macros (e.g. NAME or NAME=value).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=Path,
        help="Extra include directories for glslc.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=Path,
        help="Explicit shader source path. If omitted we scan SOURCE_DIR for *.comp.",
    )
    parser.add_argument(
        "--copy-dir",
        type=Path,
        help="Mirror compiled SPIR-V files into this directory (structure is preserved).",
    )
    parser.add_argument(
        "--depdir",
        type=Path,
        help="Directory for generated depfiles (one per shader).",
    )
    parser.add_argument("--debug", action="store_true", help="Pass -g to glslc.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    includes = [Path(p).resolve() for p in args.include]
    sources = [Path(p).resolve() for p in args.source]
    if not sources:
        sources = sorted(source_dir.glob("**/*.comp"))
    if not sources:
        ensure_dir(output_dir)
        if args.manifest:
            write_manifest(Path(args.manifest).resolve(), [])
        return 0

    glslc = discover_glslc(args.glslc)
    asset_dir = Path(args.copy_dir).resolve() if args.copy_dir else None
    dep_dir = Path(args.depdir).resolve() if args.depdir else None

    outputs = build_outputs(sources, source_dir, output_dir, asset_dir, dep_dir)

    manifest_entries = []
    for out in outputs:
        elapsed_ms = compile_one(glslc, out.source, out, includes, args.define, args.debug)
        size = out.output.stat().st_size
        manifest_entries.append(
            {
                "source": str(out.source.relative_to(source_dir)),
                "spv": str(out.output.relative_to(output_dir)),
                "size_bytes": size,
                "hash_sha256": sha256_for_file(out.output),
                "compile_ms": round(elapsed_ms, 3),
            }
        )
    if args.manifest:
        write_manifest(Path(args.manifest).resolve(), manifest_entries)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except RuntimeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(1)
