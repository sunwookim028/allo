#!/usr/bin/env python3
"""Collect an ordered RTL closure and optionally normalize it with sv2v."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path.cwd()
OUTPUTS = ROOT / "outputs"


def boolean_parameter(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be True or False")
    return value == "true"


def resolve_from_construct(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    construct = Path(os.environ["construct_path"]).expanduser().resolve()
    return (construct.parent / path).resolve()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".v", ".sv"}:
            raise ValueError(f"RTL entry is not Verilog/SystemVerilog: {path}")
        return [path.resolve()]
    if path.is_dir():
        return sorted(
            (item.resolve() for item in path.iterdir()
             if item.is_file() and item.suffix.lower() in {".v", ".sv"}),
            key=str,
        )
    raise FileNotFoundError(f"RTL manifest entry does not exist: {path}")


def read_manifest(manifest: Path, design_root: Path) -> tuple[list[Path], list[Path]]:
    selected: list[Path] = []
    excluded: list[Path] = []
    for raw in manifest.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        is_exclude = line.startswith("!")
        value = line[1:].strip() if is_exclude else line
        entry = Path(value).expanduser()
        if not entry.is_absolute():
            entry = design_root / entry
        (excluded if is_exclude else selected).extend(source_files(entry))

    excluded_set = set(excluded)
    ordered, seen = [], set()
    for path in selected:
        if path not in excluded_set and path not in seen:
            ordered.append(path)
            seen.add(path)
    if not ordered:
        raise ValueError(f"RTL manifest contains no sources: {manifest}")
    return ordered, excluded


def relative_source_path(path: Path, design_root: Path) -> Path:
    try:
        return path.relative_to(design_root)
    except ValueError:
        token = hashlib.sha256(str(path.parent).encode()).hexdigest()[:12]
        return Path("external") / token / path.name


def copy_source_closure(files: list[Path], include_dirs: list[Path], design_root: Path) -> dict[Path, Path]:
    destination = OUTPUTS / "rtl-source-package"
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    copied: dict[Path, Path] = {}

    def copy_one(source: Path) -> None:
        source = source.resolve()
        if source in copied:
            return
        relative = relative_source_path(source, design_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[source] = relative

    for path in files:
        copy_one(path)
    for directory in include_dirs:
        for pattern in ("*.vh", "*.svh"):
            for path in sorted(directory.rglob(pattern), key=str):
                copy_one(path)
    return copied


def module_names(text: str) -> list[str]:
    return re.findall(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b", text)


def synthesis_compatible_systemverilog(text: str) -> str:
    """Remove the typed-string qualifier unsupported by DC Presto."""
    return re.sub(r"\b(parameter|localparam)\s+string\s+", r"\1 ", text)


def tcl_list(values: list[str]) -> str:
    atoms = ["{" + value.replace("}", "\\}") + "}" for value in values]
    return "[list " + " ".join(atoms) + "]"


def main() -> None:
    for required in ("design_path", "manifest", "top_module", "construct_path"):
        if not os.environ.get(required) or os.environ[required] == "undefined":
            raise ValueError(f"missing parameter: {required}")

    design_root = resolve_from_construct(os.environ["design_path"])
    manifest = resolve_from_construct(os.environ["manifest"])
    if not design_root.is_dir():
        raise FileNotFoundError(f"design_path does not exist: {design_root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")

    include_dirs = []
    for value in os.environ.get("sv2v_include_dirs", ".").split(":"):
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = design_root / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"include directory does not exist: {path}")
        include_dirs.append(path)

    files, excluded = read_manifest(manifest, design_root)
    OUTPUTS.mkdir(exist_ok=True)
    copied = copy_source_closure(files, include_dirs, design_root)
    for source, relative in copied.items():
        if source.suffix.lower() not in {".v", ".sv", ".vh", ".svh"}:
            continue
        packaged = OUTPUTS / "rtl-source-package" / relative
        text = packaged.read_text(errors="replace")
        packaged.write_text(synthesis_compatible_systemverilog(text))
    source_alias = OUTPUTS / "source-rtl"
    if source_alias.is_symlink() or source_alias.is_file():
        source_alias.unlink()
    elif source_alias.is_dir():
        shutil.rmtree(source_alias)
    source_alias.symlink_to("rtl-source-package", target_is_directory=True)
    (OUTPUTS / "source-manifest.f").write_text(
        "\n".join(str(path) for path in files) + "\n"
    )

    normalize = boolean_parameter("normalize_rtl", True)
    defines = shlex.split(os.environ.get("sv2v_defines", ""))
    sv2v_bin = os.environ.get("sv2v_bin", "sv2v")
    command = [sv2v_bin]
    for define in defines:
        command.extend(["-D", define])
    for directory in include_dirs:
        command.extend(["-I", str(directory)])
    command.extend(["-v", "-w", str(OUTPUTS / "design.v")])
    command.extend(str(path) for path in files)

    log_lines = []
    if normalize:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False)
        log_lines.extend(["Running: " + shlex.join(command), completed.stdout])
        if completed.returncode:
            (OUTPUTS / "sv2v.log").write_text("\n".join(log_lines))
            raise RuntimeError(f"sv2v exited with status {completed.returncode}")
    else:
        with (OUTPUTS / "design.v").open("wb") as destination:
            for path in files:
                destination.write(path.read_bytes())
                destination.write(b"\n")
        log_lines.append("SV2V bypassed; concatenated ordered Verilog sources")

    design = OUTPUTS / "design.v"
    if not design.is_file() or design.stat().st_size == 0:
        raise RuntimeError("RTL collection produced an empty design.v")
    text = design.read_text(errors="replace")
    design.write_text(synthesis_compatible_systemverilog(text))
    text = design.read_text(errors="replace")
    modules = module_names(text)
    top = os.environ["top_module"]
    if top not in modules:
        raise ValueError(f"normalized RTL does not contain top module {top}")
    duplicates = sorted(name for name in set(modules) if modules.count(name) > 1)
    if duplicates:
        raise ValueError("normalized RTL contains duplicate modules: " + ", ".join(duplicates))

    normalized = OUTPUTS / "normalized-rtl"
    normalized.mkdir(exist_ok=True)
    shutil.copy2(design, normalized / "design.v")
    if normalize:
        rtl_filelist = ["inputs/design.v"]
        rtl_source_files = ["inputs/design.v"]
        rtl_include_dirs = []
        rtl_defines = []
    else:
        rtl_filelist = [f"+define+{define}" for define in defines]
        include_parents = []
        for source, relative in copied.items():
            if source.suffix.lower() not in {".vh", ".svh"}:
                continue
            parent = relative.parent
            if parent not in include_parents:
                include_parents.append(parent)
        rtl_filelist.extend(
            f"+incdir+inputs/rtl-source-package/{parent}"
            for parent in include_parents
        )
        rtl_filelist.extend(
            f"inputs/rtl-source-package/{copied[path]}" for path in files
        )
        rtl_source_files = [
            f"inputs/rtl-source-package/{copied[path]}" for path in files
        ]
        rtl_include_dirs = [
            f"inputs/rtl-source-package/{parent}" for parent in include_parents
        ]
        rtl_defines = defines
    (OUTPUTS / "rtl-sources.f").write_text("\n".join(rtl_filelist) + "\n")
    (OUTPUTS / "rtl-sources.tcl").write_text("\n".join([
        f"set rtl_source_files {tcl_list(rtl_source_files)}",
        f"set rtl_include_dirs {tcl_list(rtl_include_dirs)}",
        f"set rtl_defines {tcl_list(rtl_defines)}",
    ]) + "\n")
    version = "unavailable"
    if normalize:
        probe = subprocess.run([sv2v_bin, "--version"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        version = probe.stdout.strip() or "unavailable"
    metadata = {
        "schema_version": 1, "top_module": top,
        "design_root": str(design_root), "manifest": str(manifest),
        "normalize_rtl": normalize, "sv2v_bin": sv2v_bin,
        "sv2v_version": version, "defines": defines,
        "include_directories": [str(path) for path in include_dirs],
        "sources": [{"path": str(path), "packaged_path": str(copied[path]), "sha256": digest(path)} for path in files],
        "excluded_sources": [str(path) for path in excluded],
        "design_v_sha256": digest(design), "module_count": len(modules),
    }
    (OUTPUTS / "rtl-collection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUTPUTS / "sv2v.log").write_text("\n".join(log_lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
