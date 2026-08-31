#!/usr/bin/env python3
"""Package user-owned SystemVerilog testbench sources and runtime collateral."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path


ROOT = Path.cwd()
OUTPUTS = ROOT / "outputs"
PACKAGE = OUTPUTS / "testbench"
FILES = PACKAGE / "files"
HDL_SUFFIXES = {".v", ".sv"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git"}


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


def expand_entry(path: Path, *, hdl_only: bool) -> list[Path]:
    if path.is_file():
        if hdl_only and path.suffix.lower() not in HDL_SUFFIXES:
            raise ValueError(f"testbench source is not Verilog/SystemVerilog: {path}")
        return [path.resolve()]
    if path.is_dir():
        return sorted(
            (item.resolve() for item in path.rglob("*")
             if item.is_file()
             and not any(part in SKIP_PARTS for part in item.parts)
             and (not hdl_only or item.suffix.lower() in HDL_SUFFIXES)),
            key=str,
        )
    raise FileNotFoundError(f"testbench entry does not exist: {path}")


def resolve_entry(value: str, testbench_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (testbench_root / path).resolve()


def read_manifest(path: Path, testbench_root: Path) -> tuple[list[Path], list[Path]]:
    sources, data, excluded = [], [], set()
    records = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("!"):
            excluded.update(expand_entry(resolve_entry(line[1:].strip(), testbench_root), hdl_only=False))
            continue
        if line.startswith("@data "):
            records.append((data, resolve_entry(line[6:].strip(), testbench_root), False))
        else:
            records.append((sources, resolve_entry(line, testbench_root), True))
    for destination, entry, hdl_only in records:
        destination.extend(item for item in expand_entry(entry, hdl_only=hdl_only) if item not in excluded)
    return deduplicate(sources), deduplicate(data)


def deduplicate(paths: list[Path]) -> list[Path]:
    result, seen = [], set()
    for path in paths:
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def packaged_relative(path: Path, testbench_root: Path) -> Path:
    try:
        return path.relative_to(testbench_root)
    except ValueError:
        token = hashlib.sha256(str(path.parent).encode()).hexdigest()[:12]
        return Path("external") / token / path.name


def main() -> None:
    for required in ("construct_path", "testbench_path", "testbench_top", "dut_instance"):
        if not os.environ.get(required) or os.environ[required] == "undefined":
            raise ValueError(f"missing parameter: {required}")
    testbench_root = resolve_from_construct(os.environ["testbench_path"])
    if not testbench_root.is_dir():
        raise FileNotFoundError(f"testbench_path does not exist: {testbench_root}")

    consume_upstream = os.environ.get("consume_upstream_testbench", "False").lower() == "true"
    manifest_value = os.environ.get("testbench_manifest", "").strip()
    main_value = os.environ.get("testbench_file", "").strip()
    if consume_upstream:
        main_file = Path("inputs/testbench.sv").resolve()
        if not main_file.is_file():
            raise FileNotFoundError("upstream testbench did not publish testbench.sv")
        sources = [main_file]
        data = [path.resolve() for path in Path("inputs").iterdir()
                if path.is_file() and path.resolve() != main_file]
        testbench_root = Path("inputs").resolve()
        manifest = None
    elif manifest_value:
        manifest = resolve_from_construct(manifest_value)
        if not manifest.is_file():
            raise FileNotFoundError(f"testbench manifest does not exist: {manifest}")
        sources, data = read_manifest(manifest, testbench_root)
    else:
        if not main_value or main_value == "undefined":
            raise ValueError("testbench_file is required when testbench_manifest is empty")
        main_file = resolve_entry(main_value, testbench_root)
        sources = expand_entry(main_file, hdl_only=True)
        # Simple mode packages neighboring collateral without compiling every
        # HDL file. Multi-file compile order remains an explicit manifest job.
        data = [path for path in expand_entry(testbench_root, hdl_only=False) if path not in sources]
        manifest = None
    if not sources:
        raise ValueError("testbench source list is empty")

    include_dirs = []
    for value in os.environ.get("testbench_include_dirs", ".").split(":"):
        if not value:
            continue
        path = resolve_entry(value, testbench_root)
        if not path.is_dir():
            raise FileNotFoundError(f"testbench include directory does not exist: {path}")
        include_dirs.append(path)
        for pattern in ("*.vh", "*.svh"):
            data.extend(path.rglob(pattern))
    data = deduplicate([path.resolve() for path in data if path.is_file()])

    shutil.rmtree(PACKAGE, ignore_errors=True)
    FILES.mkdir(parents=True)
    all_files = deduplicate(sources + data)
    packaged = {}
    for source in all_files:
        relative = packaged_relative(source, testbench_root)
        destination = FILES / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        packaged[source] = relative

    filelist = []
    for define in shlex.split(os.environ.get("testbench_defines", "")):
        filelist.append(f"+define+{define}")
    for directory in include_dirs:
        relative = packaged_relative(directory, testbench_root)
        filelist.append(f"+incdir+inputs/testbench/files/{relative}")
    filelist.extend(f"inputs/testbench/files/{packaged[path]}" for path in sources)
    (PACKAGE / "testbench.f").write_text("\n".join(filelist) + "\n")

    args_value = os.environ.get("simulation_args_file", "").strip()
    compile_args = ""
    if consume_upstream and Path("inputs/design.args").is_file():
        compile_args = Path("inputs/design.args").read_text()
    elif args_value:
        args_path = resolve_from_construct(args_value)
        if not args_path.is_file():
            raise FileNotFoundError(f"simulation args file does not exist: {args_path}")
        compile_args = args_path.read_text()
    (PACKAGE / "testbench-compile.args").write_text(compile_args)
    (PACKAGE / "testbench-runtime.args").write_text("")

    contract = {
        "schema_version": 1,
        "testbench_top": os.environ["testbench_top"],
        "dut_instance": os.environ["dut_instance"],
        "pass_marker": os.environ.get("pass_marker", "TEST_PASS"),
        "failure_marker": os.environ.get("failure_marker", "TEST_FAIL"),
        "simulation_timeout_seconds": int(os.environ.get("simulation_timeout_seconds", "3600")),
        "source_filelist": "inputs/testbench/testbench.f",
        "compile_args_file": "inputs/testbench/testbench-compile.args",
        "runtime_args_file": "inputs/testbench/testbench-runtime.args",
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "testbench-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    report = {
        "schema_version": 1, "node": "testbench-collector", "status": "passed",
        "testbench_root": str(testbench_root),
        "manifest": str(manifest) if manifest else None,
        "source_count": len(sources), "data_file_count": len(data),
        "sources": [{"path": str(path), "packaged_path": str(packaged[path]), "sha256": digest(path)} for path in sources],
        "data": [{"path": str(path), "packaged_path": str(packaged[path]), "sha256": digest(path)} for path in data],
    }
    (OUTPUTS / "testbench-collection-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
