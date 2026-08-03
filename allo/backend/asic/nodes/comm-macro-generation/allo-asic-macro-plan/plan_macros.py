#!/usr/bin/env python3
"""Select profitable RTL macro classes and package canonical representatives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
MODULE_RE = re.compile(rf"(?<![A-Za-z0-9_$])module\s+(?P<name>{IDENT})\b")
ENDMODULE_RE = re.compile(r"(?<![A-Za-z0-9_$])endmodule\b")


@dataclass(frozen=True)
class ModuleBlock:
    name: str
    start: int
    end: int
    text: str


def module_blocks(text: str) -> dict[str, ModuleBlock]:
    """Return non-nested Verilog module definitions with source spans."""
    blocks: dict[str, ModuleBlock] = {}
    cursor = 0
    while True:
        match = MODULE_RE.search(text, cursor)
        if match is None:
            break
        end_match = ENDMODULE_RE.search(text, match.end())
        if end_match is None:
            raise ValueError(f"module {match.group('name')} has no endmodule")
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start]
        if re.fullmatch(r"\s*(?:\(\*.*?\*\)\s*)*", prefix, re.DOTALL):
            start = line_start
        end = end_match.end()
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == "\n":
            end += 1
        name = match.group("name")
        if name in blocks:
            raise ValueError(f"duplicate module definition: {name}")
        blocks[name] = ModuleBlock(name, start, end, text[start:end])
        cursor = end
    return blocks


def balanced_parentheses(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated parenthesized list")


def port_names(block: ModuleBlock) -> list[str]:
    declaration = MODULE_RE.search(block.text)
    if declaration is None:
        raise ValueError(f"cannot parse declaration for {block.name}")
    opening = block.text.find("(", declaration.end())
    if opening < 0:
        raise ValueError(f"module {block.name} has no port list")
    closing = balanced_parentheses(block.text, opening)
    raw = block.text[opening + 1 : closing]
    ports = [item.strip() for item in raw.split(",") if item.strip()]
    if not all(re.fullmatch(IDENT, port) for port in ports):
        raise ValueError(f"module {block.name} does not use a simple non-ANSI port list")
    return ports


def port_declarations(block: ModuleBlock, ports: list[str]) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for statement in re.findall(r"\b(?:input|output|inout)\b[^;]*;", block.text):
        for port in ports:
            if re.search(rf"\b{re.escape(port)}\b", statement):
                normalized = re.sub(r"\breg\b", "wire", statement)
                declarations[port] = normalized.strip()
    missing = [port for port in ports if port not in declarations]
    if missing:
        raise ValueError(f"module {block.name} lacks declarations for ports {missing}")
    return declarations


def normalized_declaration(declaration: str) -> str:
    return re.sub(
        r"\b(?:wire|reg)\b", "", re.sub(r"\s+", " ", declaration)
    ).strip()


STREAM_SUFFIXES = (
    "_num_data_valid",
    "_fifo_cap",
    "_empty_n",
    "_full_n",
    "_ap_vld",
    "_dout",
    "_read",
    "_din",
    "_write",
)


def stream_bundle_root(port: str) -> str | None:
    """Return the Vitis interface-bundle root for a generated RTL port."""
    for suffix in STREAM_SUFFIXES:
        if port.endswith(suffix):
            return port[: -len(suffix)]
    return None


def stream_bundles(block: ModuleBlock) -> list[dict[str, object]]:
    """Find Vitis FIFO/stream bundles in declaration order.

    FIFO inputs are rooted at ``*_dout`` and FIFO outputs at ``*_din``.  Vitis
    also emits scalar/AXIS-like outputs as ``value,value_ap_vld``; those are
    treated as an output bundle as well.
    """
    ports = port_names(block)
    declarations = port_declarations(block, ports)
    grouped: dict[str, list[str]] = {}
    first: dict[str, int] = {}
    for index, port in enumerate(ports):
        root = stream_bundle_root(port)
        if root is None and f"{port}_ap_vld" in ports:
            root = port
        if root is None:
            continue
        grouped.setdefault(root, []).append(port)
        first.setdefault(root, index)

    bundles = []
    for root in sorted(grouped, key=first.get):
        names = grouped[root]
        if f"{root}_dout" in names:
            direction = "in"
        elif f"{root}_din" in names or f"{root}_ap_vld" in names:
            direction = "out"
        else:
            continue
        # Retain only actual top-level ports and validate their declarations.
        bundles.append(
            {
                "root": root,
                "direction": direction,
                "rtl_ports": [name for name in names if name in declarations],
            }
        )
    return bundles


def axis_hint(stream: str) -> str | None:
    lowered = stream.lower()
    if any(token in lowered for token in ("horizontal", "hori", "east", "west")):
        return "horizontal"
    if any(token in lowered for token in ("vertical", "vert", "north", "south")):
        return "vertical"
    return None


def opposite(side: str) -> str:
    return {"N": "S", "S": "N", "E": "W", "W": "E"}[side]


def declaration_width(declaration: str) -> int:
    match = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", declaration)
    if match is None:
        return 1
    return abs(int(match.group(1)) - int(match.group(2))) + 1


def auxiliary_group(port: str) -> str:
    """Keep each AXI channel together while balancing external-interface pins."""
    match = re.match(r"^(m_axi_[A-Za-z0-9$]+)_(AW|AR|W|R|B)", port)
    if match is not None:
        return f"{match.group(1)}_{match.group(2)}"
    match = re.match(r"^(s_axi_[A-Za-z0-9$]+)_(AW|AR|W|R|B)", port)
    if match is not None:
        return f"{match.group(1)}_{match.group(2)}"
    return port


def balance_auxiliary_pins(
    auxiliary: list[str],
    declarations: dict[str, str],
    stream_sides: set[str],
    initial_loads: dict[str, int],
) -> dict[str, str]:
    """Assign auxiliary interface groups to lightly loaded non-stream sides."""
    candidates = [side for side in ("W", "N", "S", "E") if side not in stream_sides]
    if not candidates:
        candidates = ["W", "N", "S", "E"]
    groups: dict[str, list[str]] = {}
    for port in auxiliary:
        groups.setdefault(auxiliary_group(port), []).append(port)
    weighted_groups = sorted(
        (
            sum(declaration_width(declarations[port]) for port in ports),
            name,
            ports,
        )
        for name, ports in groups.items()
    )
    loads = dict(initial_loads)
    assignments = {}
    for width, _name, ports in reversed(weighted_groups):
        side = min(candidates, key=lambda item: (loads[item], candidates.index(item)))
        for port in ports:
            assignments[port] = side
        loads[side] += width
    return assignments


def graph_pin_sides(manifest: dict) -> dict[tuple[str, int], dict[str, str]]:
    """Assign a physical side to every semantic stream endpoint.

    Explicit compiler directions win.  Otherwise same-kernel PID displacement
    determines the side.  For cross-kernel edges, stream axis names select the
    axis and dataflow direction selects the sign.  The final fallback is the
    conventional west-to-east producer/consumer orientation.  Every decision
    records its method so downstream validation can reject or review fallbacks.
    """
    pes = {item["semantic_id"]: item for item in manifest.get("pe_instances", [])}
    result: dict[tuple[str, int], dict[str, str]] = {}
    for channel in manifest.get("channels", []):
        endpoints = channel.get("endpoints", [])
        for endpoint in endpoints:
            pe_id = endpoint["pe"]
            pe = pes[pe_id]
            accesses = endpoint.get("accesses", [])
            peer = next((item for item in endpoints if item["pe"] != pe_id), None)
            for access in accesses:
                ordinal = int(access["port_ordinal"])
                port = next(
                    item for item in pe.get("ports", []) if int(item["ordinal"]) == ordinal
                )
                explicit = str(port.get("desired_compass_direction", "unassigned")).upper()
                if explicit in {"N", "S", "E", "W"}:
                    side, method = explicit, "manifest_explicit"
                elif peer is not None and pes[peer["pe"]].get("kernel") == pe.get("kernel"):
                    here = list(pe.get("pid", []))
                    there = list(pes[peer["pe"]].get("pid", []))
                    if len(here) >= 2 and len(there) >= 2 and here[-1] != there[-1]:
                        side = "E" if there[-1] > here[-1] else "W"
                        method = "same_kernel_pid_column"
                    elif len(here) >= 1 and len(there) >= 1 and here[-2:] != there[-2:]:
                        # Increasing row index is physically south.
                        side = "S" if there[-2] > here[-2] else "N"
                        method = "same_kernel_pid_row"
                    else:
                        side = "E" if endpoint.get("direction") == "out" else "W"
                        method = "dataflow_fallback"
                else:
                    hint = axis_hint(channel.get("stream", ""))
                    is_out = endpoint.get("direction") == "out"
                    if hint == "vertical":
                        side = "S" if is_out else "N"
                        method = "stream_axis_hint"
                    else:
                        side = "E" if is_out else "W"
                        method = "stream_axis_hint" if hint == "horizontal" else "dataflow_fallback"
                result[(pe_id, ordinal)] = {"side": side, "method": method}
    return result


def build_pin_intent(
    manifest: dict, representative: str, canonical: ModuleBlock
) -> dict[str, object]:
    pes = {item["semantic_id"]: item for item in manifest.get("pe_instances", [])}
    if representative not in pes:
        raise ValueError(f"macro representative {representative} is absent from pe_instances")
    pe = pes[representative]
    semantic_ports = sorted(pe.get("ports", []), key=lambda item: int(item["ordinal"]))
    bundles = stream_bundles(canonical)
    selection_method = "complete_process_interface"
    if len(bundles) != len(semantic_ports):
        # Vitis may split one Allo kernel into several independently instantiated
        # pipeline processes (for example, one loader process per stream).  The
        # final manifest lists those process records in specialization order.
        # Select the corresponding same-direction semantic port by that stable
        # record ordinal.
        bundle_directions = {item["direction"] for item in bundles}
        matching_records = [
            record
            for record in pe.get("post_hls_records", [])
            if any(
                module.get("name") == canonical.name
                for module in record.get("rtl_modules", [])
            )
        ]
        all_process_records = [
            record
            for record in pe.get("post_hls_records", [])
            if record.get("rtl_modules")
        ]
        if len(bundle_directions) != 1 or len(matching_records) != 1:
            raise ValueError(
                f"cannot map split Vitis process {canonical.name} to semantic ports"
            )
        direction = next(iter(bundle_directions))
        candidates = [
            item for item in semantic_ports if item.get("direction") == direction
        ]
        record_index = all_process_records.index(matching_records[0])
        start = record_index * len(bundles)
        semantic_ports = candidates[start : start + len(bundles)]
        selection_method = "split_process_record_ordinal"
        if len(semantic_ports) != len(bundles):
            raise ValueError(
                f"split-process ordinal {record_index} is outside the {direction} "
                f"semantic interface of {canonical.name}"
            )
    side_map = graph_pin_sides(manifest)
    mapped = []
    for semantic, bundle in zip(semantic_ports, bundles):
        if semantic.get("direction") != bundle["direction"]:
            raise ValueError(
                f"stream direction mismatch for {canonical.name} ordinal "
                f"{semantic['ordinal']}: manifest={semantic.get('direction')} "
                f"rtl={bundle['direction']}"
            )
        decision = side_map[(representative, int(semantic["ordinal"]))]
        mapped.append({**semantic, **bundle, **decision})

    stream_rtl_ports = {port for item in mapped for port in item["rtl_ports"]}
    canonical_ports = port_names(canonical)
    declarations = port_declarations(canonical, canonical_ports)
    remaining = [port for port in canonical_ports if port not in stream_rtl_ports]
    control = [port for port in remaining if port.startswith("ap_")]
    auxiliary = [port for port in remaining if port not in control]
    stream_sides = {item["side"] for item in mapped}
    side_loads = {side: 0 for side in ("N", "S", "E", "W")}
    for bundle in mapped:
        side_loads[bundle["side"]] += sum(
            declaration_width(declarations[port]) for port in bundle["rtl_ports"]
        )
    side_loads["S"] += sum(declaration_width(declarations[port]) for port in control)
    auxiliary_pin_sides = balance_auxiliary_pins(
        auxiliary, declarations, stream_sides, side_loads
    )
    return {
        "schema_version": 1,
        "representative": representative,
        "module": canonical.name,
        "stream_bundles": mapped,
        "control_pins": control,
        "control_side": "S",
        "auxiliary_pins": auxiliary,
        "auxiliary_pin_sides": auxiliary_pin_sides,
        "semantic_to_rtl_method": selection_method,
    }


def write_pin_intent_tcl(path: Path, intent: dict[str, object]) -> None:
    side_ports = {side: [] for side in ("N", "S", "E", "W")}
    for bundle in intent["stream_bundles"]:
        side_ports[bundle["side"]].extend(bundle["rtl_ports"])
    side_ports[intent["control_side"]].extend(intent["control_pins"])
    for port in intent["auxiliary_pins"]:
        side_ports[intent["auxiliary_pin_sides"][port]].append(port)
    lines = ["# Generated from the Allo whole-region channel graph."]
    for side in ("N", "S", "E", "W"):
        lines.append(
            f"set allo_asic_signal_pins_{side} [list "
            + " ".join(tcl_quote(port) for port in side_ports[side])
            + "]"
        )
    path.write_text("\n".join(lines) + "\n")


D4_SIDE_MAPS = {
    "R0": {"N": "N", "E": "E", "S": "S", "W": "W"},
    "R90": {"N": "W", "E": "N", "S": "E", "W": "S"},
    "R180": {"N": "S", "E": "W", "S": "N", "W": "E"},
    "R270": {"N": "E", "E": "S", "S": "W", "W": "N"},
    "MX": {"N": "S", "E": "E", "S": "N", "W": "W"},
    "MY": {"N": "N", "E": "W", "S": "S", "W": "E"},
    "MXR90": {"N": "E", "E": "N", "S": "W", "W": "S"},
    "MYR90": {"N": "W", "E": "S", "S": "E", "W": "N"},
}


def stream_side_by_rtl_port(intent: dict[str, object]) -> dict[str, str]:
    return {
        port: bundle["side"]
        for bundle in intent["stream_bundles"]
        for port in bundle["rtl_ports"]
    }


def select_member_orientation(
    canonical_intent: dict[str, object],
    member_intent: dict[str, object],
    port_map: list[dict[str, str]],
) -> str:
    canonical_sides = stream_side_by_rtl_port(canonical_intent)
    member_sides = stream_side_by_rtl_port(member_intent)
    constraints = [
        (canonical_sides[item["canonical"]], member_sides[item["member"]])
        for item in port_map
        if item["canonical"] in canonical_sides and item["member"] in member_sides
    ]
    if not constraints:
        return "R0"
    for orientation, transform in D4_SIDE_MAPS.items():
        if all(transform[source] == target for source, target in constraints):
            return orientation
    raise ValueError(
        "equivalent RTL member has stream-side intent that is not a D4 transform "
        f"of its canonical macro: {constraints}"
    )


def validate_and_map_ports(
    canonical: ModuleBlock, member: ModuleBlock
) -> list[dict[str, str]]:
    canonical_ports = port_names(canonical)
    member_ports = port_names(member)
    if len(canonical_ports) != len(member_ports):
        raise ValueError(
            f"equivalent modules {canonical.name} and {member.name} have different "
            f"port counts ({len(canonical_ports)} != {len(member_ports)})"
        )
    canonical_decls = port_declarations(canonical, canonical_ports)
    member_decls = port_declarations(member, member_ports)
    mapping = []
    for canonical_port, member_port in zip(canonical_ports, member_ports):
        left = normalized_declaration(
            canonical_decls[canonical_port].replace(canonical_port, "PORT")
        )
        right = normalized_declaration(
            member_decls[member_port].replace(member_port, "PORT")
        )
        if left != right:
            raise ValueError(
                f"port shape mismatch in class: {canonical.name}.{canonical_port} "
                f"({left}) != {member.name}.{member_port} ({right})"
            )
        mapping.append({"canonical": canonical_port, "member": member_port})
    return mapping


def alias_wrapper(
    canonical: ModuleBlock, member: ModuleBlock, mapping: list[dict[str, str]]
) -> str:
    ports = port_names(member)
    declarations = port_declarations(member, ports)
    lines = [f"module {member.name} (", "  " + ",\n  ".join(ports), ");"]
    lines.extend(f"  {declarations[port]}" for port in ports)
    lines.append(f"  {canonical.name} canonical_macro (")
    connections = [
        f"    .{item['canonical']}({item['member']})" for item in mapping
    ]
    lines.append(",\n".join(connections))
    lines.extend(["  );", "endmodule", ""])
    return "\n".join(lines)


def dependency_closure(top: str, blocks: dict[str, ModuleBlock]) -> list[str]:
    closure: list[str] = []
    pending = [top]
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.append(name)
        body = blocks[name].text
        for candidate in sorted(blocks):
            if candidate == name or candidate in closure:
                continue
            pattern = rf"\b{re.escape(candidate)}\b\s*(?:#\s*\([^;]*?\)\s*)?{IDENT}\s*\("
            if re.search(pattern, body, re.DOTALL):
                pending.append(candidate)
    return closure


def write_constraints(path: Path, clock_period: float) -> None:
    path.write_text(
        "# Generic HLS macro constraints generated by allo-asic-macro-plan\n"
        f"create_clock -name clk -period {clock_period:g} [get_ports ap_clk]\n"
        "set nonclock_inputs [remove_from_collection [all_inputs] [get_ports ap_clk]]\n"
        "if {[sizeof_collection $nonclock_inputs] > 0} {\n"
        "  set_input_delay 0.0 -clock clk $nonclock_inputs\n"
        "}\n"
        "if {[sizeof_collection [all_outputs]] > 0} {\n"
        "  set_output_delay 0.0 -clock clk [all_outputs]\n"
        "}\n"
    )


def tcl_quote(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace("}", "\\}")
    return "{" + text + "}"


def main() -> None:
    outputs = Path("outputs")
    batch_dir = outputs / "macro-batch"
    shutil.rmtree(batch_dir, ignore_errors=True)
    batch_dir.mkdir(parents=True)
    outputs.mkdir(exist_ok=True)

    rtl_path = Path("inputs/design.v")
    manifest_path = Path("inputs/asic-manifest-final.json")
    rtl = rtl_path.read_text()
    manifest = json.loads(manifest_path.read_text())
    blocks = module_blocks(rtl)
    threshold = int(os.environ.get("min_macro_reuse", "2"))
    clock_period = float(os.environ.get("clock_period", "10.0"))
    if threshold < 1:
        raise ValueError("min_macro_reuse must be at least 1")

    summary = manifest.get("summary", {})
    if summary.get("unmatched_or_ambiguous", 0) or summary.get(
        "unjoined_post_hls_records", 0
    ):
        raise ValueError("final ASIC manifest contains unmatched or unjoined records")

    selected = []
    replacements: dict[str, str] = {}
    removed: set[str] = set()
    for group in manifest.get("macro_groups", []):
        members = group.get("members", [])
        if group.get("proof", {}).get("status") != "proven":
            continue
        if int(group.get("member_count", len(members))) < threshold:
            continue
        module_names = [member["rtl_module"] for member in members]
        missing = [name for name in module_names if name not in blocks]
        if missing:
            raise ValueError(
                f"class {group['macro_class_id']} references missing RTL modules {missing}"
            )
        canonical_name = module_names[0]
        canonical = blocks[canonical_name]
        entry_id = group["macro_class_id"]
        entry_dir = batch_dir / "entries" / entry_id
        rtl_dir = entry_dir / "rtl"
        rtl_dir.mkdir(parents=True)

        port_maps = {}
        for member_name in module_names:
            mapping = validate_and_map_ports(canonical, blocks[member_name])
            port_maps[member_name] = mapping
            removed.add(member_name)
            if member_name != canonical_name:
                replacements[member_name] = alias_wrapper(
                    canonical, blocks[member_name], mapping
                )

        dependencies = dependency_closure(canonical_name, blocks)
        design_text = "\n".join(blocks[name].text for name in dependencies)
        design_file = rtl_dir / "design.v"
        design_file.write_text(design_text.rstrip() + "\n")
        constraints_file = entry_dir / "constraints.tcl"
        write_constraints(constraints_file, clock_period)
        pin_intent = build_pin_intent(manifest, group["representative"], canonical)
        pin_intent_json = entry_dir / "pin-intent.json"
        pin_intent_tcl = entry_dir / "pin-intent.tcl"
        pin_intent_json.write_text(json.dumps(pin_intent, indent=2) + "\n")
        write_pin_intent_tcl(pin_intent_tcl, pin_intent)
        member_placements = []
        for member in members:
            member_module = member["rtl_module"]
            member_intent = build_pin_intent(
                manifest, member["semantic_id"], blocks[member_module]
            )
            member_placements.append(
                {
                    **member,
                    "orientation": select_member_orientation(
                        pin_intent, member_intent, port_maps[member_module]
                    ),
                    "stream_pin_sides": {
                        str(bundle["ordinal"]): bundle["side"]
                        for bundle in member_intent["stream_bundles"]
                    },
                }
            )
        digest = hashlib.sha256(design_file.read_bytes()).hexdigest()
        entry = {
            "id": entry_id,
            "top_module": canonical_name,
            "representative_semantic_id": group["representative"],
            "reuse_count": int(group["member_count"]),
            "rtl_hash": group["proof"]["rtl_hash"],
            "rtl_sha256": digest,
            "rtl": f"entries/{entry_id}/rtl/design.v",
            "constraints": f"entries/{entry_id}/constraints.tcl",
            "pin_intent": f"entries/{entry_id}/pin-intent.json",
            "pin_intent_tcl": f"entries/{entry_id}/pin-intent.tcl",
            "dependencies": dependencies[1:],
            "member_modules": module_names,
            "members": members,
            "member_placements": member_placements,
            "port_maps": port_maps,
        }
        (entry_dir / "entry.json").write_text(json.dumps(entry, indent=2) + "\n")
        selected.append(entry)

    if not selected:
        raise ValueError(f"no proven macro classes meet reuse threshold {threshold}")

    chunks = []
    cursor = 0
    for block in sorted(blocks.values(), key=lambda item: item.start):
        chunks.append(rtl[cursor : block.start])
        if block.name in replacements:
            chunks.append(replacements[block.name])
        elif block.name not in removed:
            chunks.append(block.text)
        cursor = block.end
    chunks.append(rtl[cursor:])
    residual = "".join(chunks)
    (outputs / "residual-design.v").write_text(residual)

    plan = {
        "schema_version": 1,
        "stage": "macro_plan",
        "top": manifest.get("top"),
        "reuse_threshold": threshold,
        "clock_period_ns": clock_period,
        "source_manifest": str(manifest_path),
        "selected_class_count": len(selected),
        "selected_instance_count": sum(entry["reuse_count"] for entry in selected),
        "entries": selected,
    }
    index_text = json.dumps(plan, indent=2) + "\n"
    (batch_dir / "index.json").write_text(index_text)
    (outputs / "macro-plan.json").write_text(index_text)
    shutil.copy2("inputs/asic-manifest-final.json", batch_dir / "asic-manifest-final.json")
    shutil.copy2("inputs/build-metadata.json", batch_dir / "build-metadata.json")

    tcl_lines = [
        f"set allo_asic_macro_plan_schema_version {plan['schema_version']}",
        f"set allo_asic_macro_reuse_threshold {threshold}",
        "set allo_asic_macro_class_ids [list "
        + " ".join(tcl_quote(entry["id"]) for entry in selected)
        + "]",
    ]
    for entry in selected:
        key = entry["id"]
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", key):
            raise ValueError(f"macro class ID is not Tcl-array safe: {key}")
        tcl_lines.extend(
            [
                f"set allo_asic_macro_top({key}) {tcl_quote(entry['top_module'])}",
                f"set allo_asic_macro_reuse({key}) {entry['reuse_count']}",
                f"set allo_asic_macro_rtl({key}) {tcl_quote(entry['rtl'])}",
            ]
        )
    (outputs / "macro-plan.tcl").write_text("\n".join(tcl_lines) + "\n")
    log = (
        f"Selected {len(selected)} proven macro classes covering "
        f"{plan['selected_instance_count']} instances at reuse threshold {threshold}.\n"
    )
    (outputs / "macro-plan.log").write_text(log)
    print(log, end="")


if __name__ == "__main__":
    main()
