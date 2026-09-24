#!/usr/bin/env python3
"""Validate, package, or generate SRAM collateral behind one contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


CONTRACT_VERSION = 1
MODES = {"none", "provided", "bypass", "generate"}
VIEW_PATTERNS = {
    "verilog": ("*.v", "*.sv"),
    "liberty": ("*.lib",),
    "database": ("*.db",),
    "lef": ("*.lef",),
    "gds": ("*.gds", "*.gds.gz"),
    "spice": ("*.sp", "*.spi", "*.cdl"),
}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
OUTPUTS = Path("outputs")
PACKAGE = OUTPUTS / "srams"


def parameter(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def construct_dir() -> Path:
    value = parameter("construct_path", "undefined")
    if value == "undefined":
        raise ValueError("construct_path must be set for this SRAM mode")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"construct_path does not exist: {path}")
    return path.parent


def resolve_design_path(value: str, parameter_name: str) -> Path:
    if not value or value == "undefined":
        raise ValueError(f"{parameter_name} must be set")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = construct_dir() / path
    return path.resolve()


def find_executable(value: str, parameter_name: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise FileNotFoundError(f"{parameter_name} is not executable: {resolved}")
        return str(resolved)
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(f"Required executable for {parameter_name} was not found: {value}")
    return resolved


def load_generation_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read sram_manifest") from error

    if not path.is_file():
        raise FileNotFoundError(f"sram_manifest does not exist: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("srams"), list):
        raise ValueError(f"sram_manifest must contain an 'srams' list: {path}")
    if not data["srams"]:
        raise ValueError("generate mode requires at least one SRAM manifest entry")

    seen: set[str] = set()
    for index, entry in enumerate(data["srams"]):
        if not isinstance(entry, dict):
            raise ValueError(f"SRAM manifest entry {index} must be a mapping")
        for key in ("name", "word_size", "num_words"):
            if key not in entry:
                raise ValueError(f"SRAM manifest entry {index} is missing {key}")
        name = str(entry["name"])
        if not NAME.fullmatch(name):
            raise ValueError(f"Invalid SRAM name in manifest: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate SRAM name in manifest: {name}")
        seen.add(name)
        for key in ("word_size", "num_words", "num_banks", "words_per_row", "write_size",
                    "num_rw_ports", "num_r_ports", "num_w_ports"):
            if key in entry and int(entry[key]) < (0 if key.startswith("num_") and key.endswith("ports") else 1):
                raise ValueError(f"{name}.{key} has an invalid value: {entry[key]}")
    return data


def matching_files(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(path for path in directory.glob(pattern) if path.is_file())
    return sorted(matches)


def discover_macro_names(package: Path, expected: list[str] | None = None) -> list[str]:
    if expected is not None:
        return expected
    return sorted(
        child.name for child in package.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


def validate_package(package: Path, expected: list[str] | None = None) -> list[dict[str, Any]]:
    if not package.is_dir():
        raise FileNotFoundError(f"SRAM package directory does not exist: {package}")
    names = discover_macro_names(package, expected)
    if not names:
        raise ValueError(f"SRAM package contains no macro directories: {package}")

    entries: list[dict[str, Any]] = []
    for name in names:
        if not NAME.fullmatch(name):
            raise ValueError(f"Invalid SRAM directory name: {name!r}")
        macro_dir = package / name
        if not macro_dir.is_dir():
            raise FileNotFoundError(f"Missing SRAM directory: {macro_dir}")
        views: dict[str, list[str]] = {}
        missing: list[str] = []
        for view, patterns in VIEW_PATTERNS.items():
            files = matching_files(macro_dir, patterns)
            if not files:
                missing.append(view)
            views[view] = [str(path.relative_to(package)) for path in files]
        if missing:
            raise ValueError(f"SRAM {name} is missing required views: {', '.join(missing)}")
        entries.append({"name": name, "views": views})
    return entries


def generation_inputs() -> tuple[Path, dict[str, Any]]:
    manifest_path = resolve_design_path(parameter("sram_manifest", "rtl/sram_manifest.yml"), "sram_manifest")
    return manifest_path, load_generation_manifest(manifest_path)


def openram_preflight() -> None:
    python_bin = find_executable(parameter("python_bin", "python"), "python_bin")
    subprocess.run([python_bin, "-c", "import yaml"], check=True, capture_output=True, text=True)
    find_executable("bash", "bash")
    find_executable("lc_shell", "lc_shell")
    script = parameter("openram_script", "")
    if script:
        resolved = resolve_design_path(script, "openram_script")
        if not resolved.is_file():
            raise FileNotFoundError(f"openram_script does not exist: {resolved}")
    else:
        probe = "import openram,pathlib; p=pathlib.Path(openram.__file__).resolve().parent/'sram_compiler.py'; assert p.is_file(), p"
        subprocess.run([python_bin, "-c", probe], check=True, capture_output=True, text=True)


def preflight() -> tuple[str, dict[str, Any]]:
    mode = parameter("sram_mode", "none").lower()
    if mode not in MODES:
        raise ValueError(f"sram_mode must be one of {sorted(MODES)}, got {mode!r}")

    context: dict[str, Any] = {}
    if mode == "provided":
        source = resolve_design_path(parameter("provided_sram_path", "undefined"), "provided_sram_path")
        validate_package(source)
        context["source"] = source
    elif mode == "bypass":
        source = Path("inputs/srams").resolve()
        validate_package(source)
        context["source"] = source
    elif mode == "generate":
        method = parameter("generate_method", "openram").lower()
        if method != "openram":
            raise ValueError(f"Unsupported generate_method: {method!r}; currently supported: openram")
        manifest_path, manifest = generation_inputs()
        openram_preflight()
        context.update({"manifest_path": manifest_path, "manifest": manifest, "generate_method": method})
    return mode, context


def reset_package() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir()


def copy_package(source: Path) -> None:
    reset_package()
    shutil.copytree(source, PACKAGE, dirs_exist_ok=True)


def run_openram(context: dict[str, Any]) -> None:
    reset_package()
    env = os.environ.copy()
    script = parameter("openram_script", "")
    if script:
        env["openram_script"] = str(resolve_design_path(script, "openram_script"))
    subprocess.run(["bash", "run_openram.sh"], env=env, check=True)


def add_manifest_parameters(entries: list[dict[str, Any]], manifest: dict[str, Any] | None) -> None:
    if not manifest:
        return
    parameters = {str(item["name"]): item for item in manifest["srams"]}
    for entry in entries:
        entry["parameters"] = parameters[entry["name"]]


def publish(mode: str, entries: list[dict[str, Any]], context: dict[str, Any]) -> None:
    num_srams = len(entries)
    contract = {
        "schema": "sram-collateral-contract",
        "schema_version": CONTRACT_VERSION,
        "num_srams": num_srams,
        "srams": entries,
    }
    metadata: dict[str, Any] = {
        "schema": "sram-collateral-metadata",
        "schema_version": CONTRACT_VERSION,
        "mode": mode,
        "num_srams": num_srams,
    }
    if mode in {"provided", "bypass"}:
        metadata["source"] = str(context["source"])
    elif mode == "generate":
        metadata["generation"] = {
            "method": context["generate_method"],
            "manifest": str(context["manifest_path"]),
            "technology": parameter("tech_name", "freepdk45"),
            "process_corner": parameter("process_corner", "TT"),
            "supply_voltage": parameter("supply_voltage", "1.1"),
            "temperature": parameter("temperature", "25"),
        }

    (OUTPUTS / "sram-contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (OUTPUTS / "sram-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    lines = [
        "SRAM collateral summary",
        f"mode: {mode}",
        f"num_srams: {num_srams}",
    ]
    for entry in entries:
        lines.append(f"- {entry['name']}")
        for view, files in entry["views"].items():
            lines.append(f"    {view}: {len(files)} file(s)")
    (OUTPUTS / "sram-info.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    try:
        mode, context = preflight()
        if args.preflight:
            print(f"SRAM preflight passed for mode={mode}")
            return

        manifest = context.get("manifest")
        if mode == "none":
            reset_package()
            entries: list[dict[str, Any]] = []
        elif mode in {"provided", "bypass"}:
            copy_package(context["source"])
            entries = validate_package(PACKAGE)
        else:
            run_openram(context)
            expected = [str(item["name"]) for item in manifest["srams"]]
            entries = validate_package(PACKAGE, expected)
            add_manifest_parameters(entries, manifest)
        publish(mode, entries, context)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
