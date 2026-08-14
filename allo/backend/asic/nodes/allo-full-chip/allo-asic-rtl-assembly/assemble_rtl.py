#!/usr/bin/env python3
"""Replace selected PE instances with canonical hardened-macro interfaces."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
MODULE_RE = re.compile(rf"(?<![A-Za-z0-9_$])module\s+(?P<name>{IDENT})\b")
ENDMODULE_RE = re.compile(r"(?<![A-Za-z0-9_$])endmodule\b")


def module_blocks(text: str) -> list[dict]:
    blocks = []
    cursor = 0
    while match := MODULE_RE.search(text, cursor):
        end = ENDMODULE_RE.search(text, match.end())
        if end is None:
            raise ValueError(f"module {match.group('name')} has no endmodule")
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start]
        if re.fullmatch(r"\s*(?:\(\*.*?\*\)\s*)*", prefix, re.DOTALL):
            start = line_start
        blocks.append(
            {
                "name": match.group("name"),
                "start": start,
                "end": end.end(),
                "text": text[start : end.end()],
            }
        )
        cursor = end.end()
    return blocks


def closing_parenthesis(text: str, opening: int) -> int:
    depth = 0
    quote = None
    escaped = False
    line_comment = False
    block_comment = False
    index = opening
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
        elif block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and next_char == "/":
            line_comment = True
            index += 1
        elif char == "/" and next_char == "*":
            block_comment = True
            index += 1
        elif char == '"':
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("unterminated instance port list")


def rewrite_instances(body: str, replacement_by_module: dict[str, dict]) -> tuple[str, list[dict]]:
    candidates = sorted(replacement_by_module, key=len, reverse=True)
    if not candidates:
        return body, []
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_$])(?P<module>{'|'.join(re.escape(x) for x in candidates)})"
        rf"\s+(?P<instance>{IDENT})\s*(?P<opening>\()"
    )
    edits = []
    records = []
    cursor = 0
    while match := pattern.search(body, cursor):
        source = match.group("module")
        detail = replacement_by_module[source]
        opening = match.start("opening")
        closing = closing_parenthesis(body, opening)
        connections = body[opening + 1 : closing]
        port_map = detail["member_to_canonical_ports"]
        seen = []

        def rename_port(port_match: re.Match) -> str:
            member = port_match.group("port")
            if member not in port_map:
                raise ValueError(
                    f"instance {match.group('instance')} of {source} uses unmapped port {member}"
                )
            seen.append(member)
            return "." + port_map[member] + port_match.group("spacing") + "("

        explicit = detail.get("explicit_canonical_connections")
        if explicit:
            renamed = ",\n".join(
                f".{port}({expression})" for port, expression in explicit.items()
            )
            seen = list(explicit)
        else:
            renamed = re.sub(
                rf"\.(?P<port>{IDENT})(?P<spacing>\s*)\(", rename_port, connections
            )
            if not seen:
                raise ValueError(
                    f"instance {match.group('instance')} of {source} is not named-port RTL"
                )
            if len(set(port_map[name] for name in seen)) != len(seen):
                raise ValueError(f"canonical port collision while rewriting {source}")
        replacement = (
            detail["canonical_module"]
            + body[match.end("module") : match.start("instance")]
            + detail["stable_instance_name"]
            + body[match.end("instance") : opening + 1]
            + renamed
            + ")"
        )
        edits.append((match.start(), closing + 1, replacement))
        records.append(
            {
                "source_module": source,
                "canonical_module": detail["canonical_module"],
                "instance_name": match.group("instance"),
                "stable_instance_name": detail["stable_instance_name"],
                "rewritten_port_count": len(seen),
            }
        )
        cursor = closing + 1
    for start, end, replacement in reversed(edits):
        body = body[:start] + replacement + body[end:]
    return body, records


def remove_instances(body: str, removals: list[dict]) -> tuple[str, list[dict]]:
    edits = []
    removed = []
    for item in removals:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_$]){re.escape(item['fifo_module'])}\s+"
            rf"{re.escape(item['fifo_instance'])}\s*\("
        )
        matches = list(pattern.finditer(body))
        if len(matches) != 1:
            raise ValueError(
                f"expected one folded FIFO instance {item['fifo_module']} "
                f"{item['fifo_instance']}, found {len(matches)}"
            )
        match = matches[0]
        opening = body.find("(", match.start())
        closing = closing_parenthesis(body, opening)
        end = closing + 1
        while end < len(body) and body[end].isspace():
            end += 1
        if end >= len(body) or body[end] != ";":
            raise ValueError(f"folded FIFO {item['fifo_instance']} lacks semicolon")
        edits.append((match.start(), end + 1, ""))
        removed.append(item)
    for start, end, replacement in sorted(edits, reverse=True):
        body = body[:start] + replacement + body[end:]
    return body, removed


def interface_stub(block: str, module_name: str) -> str:
    declaration = MODULE_RE.search(block)
    opening = block.find("(", declaration.end())
    closing = closing_parenthesis(block, opening)
    header = block[declaration.start() : closing + 1] + ";\n"
    declarations = re.findall(r"(?m)^\s*(?:input|output|inout)\b[^;]*;", block)
    if not declarations:
        raise ValueError(f"cannot extract interface declarations for {module_name}")
    return header + "\n".join(item.strip() for item in declarations) + "\nendmodule\n"


def tcl_quote(value: object) -> str:
    return "{" + str(value).replace("\\", "\\\\").replace("}", "\\}") + "}"


def module_port_names(block: str) -> set[str]:
    """Return the declared port names from an ANSI or non-ANSI module header."""
    declaration = MODULE_RE.search(block)
    cursor = declaration.end()
    parameter_marker = re.match(r"\s*#", block[cursor:])
    if parameter_marker:
        parameter_opening = block.find("(", cursor + parameter_marker.end())
        cursor = closing_parenthesis(block, parameter_opening) + 1
    opening = block.find("(", cursor)
    closing = closing_parenthesis(block, opening)
    names = set()
    for item in block[opening + 1 : closing].split(","):
        identifiers = re.findall(IDENT, item)
        if identifiers:
            names.add(identifiers[-1])
    return names


def main() -> None:
    inputs = Path("inputs")
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    source = (inputs / "design.v").read_text()
    plan = json.loads((inputs / "assembly-plan.json").read_text())
    registry = json.loads((inputs / "macro-registry.json").read_text())
    configured_backend = os.environ.get("backend")
    backend = configured_backend or plan.get("backend", "vitis")
    if backend not in {"vitis", "catapult"}:
        raise ValueError(f"unsupported full-chip RTL assembly backend {backend!r}")
    plan_backend = plan.get("backend")
    if plan_backend is not None and plan_backend != backend:
        raise ValueError(
            f"RTL assembly backend {backend!r} does not match assembly plan "
            f"backend {plan_backend!r}"
        )
    replacements = plan.get("replacements", [])
    replacement_by_module = {}
    for item in replacements:
        source_module = item["source_module"]
        existing = replacement_by_module.get(source_module)
        if existing is not None and (
            existing["canonical_module"] != item["canonical_module"]
            or existing["member_to_canonical_ports"] != item["member_to_canonical_ports"]
        ):
            raise ValueError(f"inconsistent replacement records for {source_module}")
        replacement_by_module[source_module] = item
    removed = set(replacement_by_module)
    blocks = module_blocks(source)
    block_by_name = {item["name"]: item for item in blocks}
    if len(block_by_name) != len(blocks):
        raise ValueError("normalized RTL contains duplicate module definitions")
    missing = removed - set(block_by_name)
    if missing:
        raise ValueError(f"replacement definitions are missing: {sorted(missing)}")

    pieces = []
    rewritten = []
    removed_fifos = []
    fifo_removals_by_parent = {}
    for replacement in replacements:
        for fifo in replacement.get("folded_fifos", []):
            fifo_removals_by_parent.setdefault(fifo["parent_module"], []).append(fifo)
    cursor = 0
    for block in blocks:
        pieces.append(source[cursor : block["start"]])
        if block["name"] not in removed:
            body = block["text"]
            body, removed_here = remove_instances(
                body, fifo_removals_by_parent.get(block["name"], [])
            )
            removed_fifos.extend(removed_here)
            updated, records = rewrite_instances(body, replacement_by_module)
            pieces.append(updated)
            for record in records:
                record["parent_module"] = block["name"]
            rewritten.extend(records)
        cursor = block["end"]
    pieces.append(source[cursor:])
    assembled = "".join(pieces)
    expected_instances = len(replacements)
    if len(rewritten) != expected_instances:
        raise ValueError(
            f"rewrote {len(rewritten)} instances but plan requires {expected_instances}"
        )
    expected_fifos = int(plan.get("folded_fifo_count", 0))
    if len(removed_fifos) != expected_fifos:
        raise ValueError(
            f"removed {len(removed_fifos)} FIFO instances but plan requires "
            f"{expected_fifos}"
        )
    surviving = {item["name"] for item in module_blocks(assembled)}
    if surviving & removed:
        raise ValueError(f"replaced definitions survived assembly: {sorted(surviving & removed)}")
    (outputs / "assembled-design.v").write_text(assembled)
    assembled_blocks = {
        item["name"]: item["text"] for item in module_blocks(assembled)
    }
    top_module = plan["top_module"]
    if top_module not in assembled_blocks:
        raise ValueError(f"assembled RTL does not define top module {top_module!r}")
    clock_port = {"vitis": "ap_clk", "catapult": "clk"}[backend]
    top_ports = module_port_names(assembled_blocks[top_module])
    if clock_port not in top_ports:
        raise ValueError(
            f"{backend} backend requires top-level clock port {clock_port!r}; "
            f"top module {top_module!r} ports are {sorted(top_ports)}"
        )

    macros_by_id = {item["macro_class_id"]: item for item in registry.get("macros", [])}
    stubs = []
    collateral_classes = []
    for class_plan in plan.get("classes", []):
        class_id = class_plan["macro_class_id"]
        macro = macros_by_id[class_id]
        canonical = class_plan["canonical_module"]
        if canonical in block_by_name:
            canonical_block = block_by_name[canonical]["text"]
        else:
            verilog_path = inputs / "macro-registry" / macro["views"]["verilog"]["path"]
            published_blocks = {
                item["name"]: item for item in module_blocks(verilog_path.read_text())
            }
            if canonical not in published_blocks:
                raise ValueError(
                    f"published functional model lacks canonical wrapper {canonical}"
                )
            canonical_block = published_blocks[canonical]["text"]
        stubs.append(interface_stub(canonical_block, canonical))
        collateral_classes.append(
            {
                "macro_class_id": class_id,
                "canonical_module": canonical,
                "reuse_count": class_plan["reuse_count"],
                "views": macro["views"],
                "lef_symmetry": macro.get("lef_symmetry", []),
            }
        )
    (outputs / "macro-interface-stubs.v").write_text("\n".join(stubs))
    collateral = {
        "schema_version": 1,
        "stage": "full_chip_rtl_assembly",
        "top_module": plan["top_module"],
        "backend": backend,
        "clock_port": clock_port,
        "bypass_macro_generation": bool(
            plan.get("bypass_macro_generation", False)
        ),
        "implementation_style": plan.get(
            "implementation_style", "hierarchical_macros"
        ),
        "macro_classes": collateral_classes,
        "removed_module_definitions": sorted(removed),
        "rewritten_instances": rewritten,
        "removed_folded_fifos": removed_fifos,
        "synthesis_policy": (
            "read assembled-design.v and link canonical macro DB views; do not read "
            "macro-interface-stubs.v as an implementation"
        ),
    }
    (outputs / "macro-collateral.json").write_text(json.dumps(collateral, indent=2) + "\n")
    db_paths = []
    lib_paths = []
    lef_paths = []
    gds_paths = []
    verilog_paths = []
    for item in collateral_classes:
        views = item["views"]
        resolve = lambda key: "inputs/macro-registry/" + views[key]["path"]
        db_paths.append(resolve("db"))
        lib_paths.append(resolve("liberty"))
        lef_paths.append(resolve("lef"))
        gds_paths.append(resolve("gds"))
        verilog_paths.append(resolve("verilog"))
    tcl = [
        "# Registry-relative paths assume the standard macro-registry input name.",
        "set allo_asic_bypass_macro_generation "
        + ("1" if collateral["bypass_macro_generation"] else "0"),
        "set allo_asic_macro_modules [list "
        + " ".join(tcl_quote(item["canonical_module"]) for item in collateral_classes)
        + "]",
        "set allo_asic_macro_db_files [list " + " ".join(tcl_quote(x) for x in db_paths) + "]",
        "set allo_asic_macro_lib_files [list " + " ".join(tcl_quote(x) for x in lib_paths) + "]",
        "set allo_asic_macro_lef_files [list " + " ".join(tcl_quote(x) for x in lef_paths) + "]",
        "set allo_asic_macro_gds_files [list " + " ".join(tcl_quote(x) for x in gds_paths) + "]",
        "set allo_asic_macro_verilog_files [list " + " ".join(tcl_quote(x) for x in verilog_paths) + "]",
    ]
    (outputs / "macro-collateral.tcl").write_text("\n".join(tcl) + "\n")
    (outputs / "macro-functional-models.f").write_text("\n".join(verilog_paths) + "\n")
    clock_period = float(os.environ.get("clock_period", "10.0"))
    if clock_period <= 0:
        raise ValueError("clock_period must be positive")
    (outputs / "constraints.tcl").write_text(
        "# Generic full-chip constraints generated by allo-asic-rtl-assembly\n"
        f"create_clock -name clk -period {clock_period:g} [get_ports {clock_port}]\n"
        "set nonclock_inputs "
        f"[remove_from_collection [all_inputs] [get_ports {clock_port}]]\n"
        "if {[sizeof_collection $nonclock_inputs] > 0} {\n"
        "  set_input_delay 0.0 -clock clk $nonclock_inputs\n"
        "}\n"
        "if {[sizeof_collection [all_outputs]] > 0} {\n"
        "  set_output_delay 0.0 -clock clk [all_outputs]\n"
        "}\n"
    )
    report = [
        "Full-chip RTL assembly",
        f"Top module: {plan['top_module']}",
        f"Backend: {backend}",
        f"Clock port: {clock_port}",
        f"Removed PE module definitions: {len(removed)}",
        f"Rewritten macro instances: {len(rewritten)}",
        f"Canonical hardened macro interfaces: {len(collateral_classes)}",
        "Result: hardened macro definitions are external and resolved from the registry",
    ]
    (outputs / "rtl-assembly-report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
