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


def main() -> None:
    inputs = Path("inputs")
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    source = (inputs / "design.v").read_text()
    plan = json.loads((inputs / "assembly-plan.json").read_text())
    registry = json.loads((inputs / "macro-registry.json").read_text())
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
    cursor = 0
    for block in blocks:
        pieces.append(source[cursor : block["start"]])
        if block["name"] not in removed:
            updated, records = rewrite_instances(block["text"], replacement_by_module)
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
    surviving = {item["name"] for item in module_blocks(assembled)}
    if surviving & removed:
        raise ValueError(f"replaced definitions survived assembly: {sorted(surviving & removed)}")
    (outputs / "assembled-design.v").write_text(assembled)

    macros_by_id = {item["macro_class_id"]: item for item in registry.get("macros", [])}
    stubs = []
    collateral_classes = []
    for class_plan in plan.get("classes", []):
        class_id = class_plan["macro_class_id"]
        macro = macros_by_id[class_id]
        canonical = class_plan["canonical_module"]
        stubs.append(interface_stub(block_by_name[canonical]["text"], canonical))
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
        "bypass_macro_generation": bool(
            plan.get("bypass_macro_generation", False)
        ),
        "implementation_style": plan.get(
            "implementation_style", "hierarchical_macros"
        ),
        "macro_classes": collateral_classes,
        "removed_module_definitions": sorted(removed),
        "rewritten_instances": rewritten,
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
        f"create_clock -name clk -period {clock_period:g} [get_ports ap_clk]\n"
        "set nonclock_inputs [remove_from_collection [all_inputs] [get_ports ap_clk]]\n"
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
        f"Removed PE module definitions: {len(removed)}",
        f"Rewritten macro instances: {len(rewritten)}",
        f"Canonical hardened macro interfaces: {len(collateral_classes)}",
        "Result: hardened macro definitions are external and resolved from the registry",
    ]
    (outputs / "rtl-assembly-report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
