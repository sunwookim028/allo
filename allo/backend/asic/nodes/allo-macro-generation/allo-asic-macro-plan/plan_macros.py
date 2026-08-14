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


@dataclass(frozen=True)
class InstanceBlock:
    module: str
    name: str
    start: int
    end: int
    connections: dict[str, str]


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


def named_connections(text: str) -> dict[str, str]:
    result = {}
    cursor = 0
    pattern = re.compile(rf"\.({IDENT})\s*\(")
    while match := pattern.search(text, cursor):
        opening = text.find("(", match.start())
        closing = balanced_parentheses(text, opening)
        name = match.group(1)
        if name in result:
            raise ValueError(f"duplicate named connection {name}")
        result[name] = text[opening + 1 : closing].strip()
        cursor = closing + 1
    return result


def module_instances(block: ModuleBlock, modules: set[str]) -> list[InstanceBlock]:
    """Return named-port instances of known modules in one module body."""
    result = []
    for module in sorted(modules, key=len, reverse=True):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_$]){re.escape(module)}\s+(?P<name>{IDENT})\s*\("
        )
        for match in pattern.finditer(block.text):
            opening = block.text.find("(", match.start())
            closing = balanced_parentheses(block.text, opening)
            end = closing + 1
            while end < len(block.text) and block.text[end].isspace():
                end += 1
            if end >= len(block.text) or block.text[end] != ";":
                continue
            result.append(
                InstanceBlock(
                    module,
                    match.group("name"),
                    match.start(),
                    end + 1,
                    named_connections(block.text[opening + 1 : closing]),
                )
            )
    return sorted(result, key=lambda item: item.start)


def replace_declared_name(declaration: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, declaration)


def wire_declaration(declaration: str, name: str) -> str:
    width = re.search(r"\[[^\]]+\]", declaration)
    return "wire " + (width.group(0) + " " if width else "") + name + ";"


def change_declaration_direction(declaration: str, direction: str) -> str:
    return re.sub(r"\b(?:input|output|inout)\b", direction, declaration, count=1)


def outer_module_for_member(pe: dict, inner_module: str) -> str:
    matches = []
    for record in pe.get("post_hls_records", []):
        if any(item.get("name") == inner_module for item in record.get("rtl_modules", [])):
            matches.extend(
                Path(item["parent_file"]).stem
                for item in record.get("rtl_instances", [])
                if item.get("parent_file")
            )
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(
            f"cannot resolve unique outer kernel module for {pe['semantic_id']} "
            f"inner module {inner_module}: {unique}"
        )
    return unique[0]


def semantic_instance_name(pe: dict) -> str:
    kernel = re.sub(r"[^A-Za-z0-9_$]+", "_", str(pe["kernel"])).strip("_")
    pid = "_".join(str(value) for value in pe.get("pid", []))
    return kernel + ("_" + pid if pid else "")


FIFO_PRODUCER_PORTS = {
    "din": "if_din",
    "full_n": "if_full_n",
    "write": "if_write",
}
FIFO_SHARED_PORTS = {
    "num_data_valid": "if_num_data_valid",
    "fifo_cap": "if_fifo_cap",
}
FIFO_CONSUMER_PORTS = {
    "dout": "if_dout",
    "empty_n": "if_empty_n",
    "read": "if_read",
}


def normalize_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def find_owned_fifo(
    root: str,
    pe_instance: InstanceBlock,
    top_instances: list[InstanceBlock],
    blocks: dict[str, ModuleBlock],
) -> InstanceBlock:
    expected = {
        fifo_port: normalize_expression(pe_instance.connections[f"{root}_{suffix}"])
        for suffix, fifo_port in FIFO_PRODUCER_PORTS.items()
    }
    matches = []
    for instance in top_instances:
        if "fifo" not in instance.module.lower():
            continue
        if all(
            normalize_expression(instance.connections.get(port, "")) == expression
            for port, expression in expected.items()
        ):
            matches.append(instance)
    if len(matches) != 1:
        raise ValueError(
            f"cannot bind output bundle {pe_instance.module}.{root} to one FIFO: "
            f"{[(item.module, item.name) for item in matches]}"
        )
    fifo = matches[0]
    required = {
        "clk", "reset", "if_read_ce", "if_write_ce", *FIFO_PRODUCER_PORTS.values(),
        *FIFO_SHARED_PORTS.values(), *FIFO_CONSUMER_PORTS.values(),
    }
    missing = sorted(required - set(fifo.connections))
    if missing:
        raise ValueError(f"FIFO {fifo.name} lacks expected ports {missing}")
    if fifo.module not in blocks:
        raise ValueError(f"FIFO module definition is absent: {fifo.module}")
    return fifo


def generate_fifo_wrapper(
    wrapper_name: str,
    pe: dict,
    outer: ModuleBlock,
    pe_instance: InstanceBlock,
    top_instances: list[InstanceBlock],
    manifest: dict,
    blocks: dict[str, ModuleBlock],
) -> tuple[str, dict, dict[str, str]]:
    """Wrap one complete kernel module and every stream FIFO it produces."""
    outer_ports = port_names(outer)
    outer_decls = port_declarations(outer, outer_ports)
    intent = build_pin_intent(manifest, pe["semantic_id"], outer, blocks)
    outgoing = [item for item in intent["stream_bundles"] if item["direction"] == "out"]
    owned = []
    removed_ports = set()
    wrapper_decls = dict(outer_decls)
    wrapper_connections = dict(pe_instance.connections)
    pe_connection_values = {
        normalize_expression(value) for value in pe_instance.connections.values()
    }
    top_ports = set(port_names(blocks[str(manifest["top"])]))

    for bundle in outgoing:
        root = str(bundle["root"])
        fifo = find_owned_fifo(root, pe_instance, top_instances, blocks)
        for control_port in ("clk", "reset"):
            if normalize_expression(fifo.connections[control_port]) not in pe_connection_values:
                raise ValueError(
                    f"FIFO {fifo.name} {control_port} does not match a producer-kernel "
                    "clock/reset connection"
                )
        externally_visible = [
            fifo.connections[fifo_port]
            for fifo_port in FIFO_CONSUMER_PORTS.values()
            if normalize_expression(fifo.connections[fifo_port]) in top_ports
        ]
        if externally_visible:
            raise ValueError(
                f"FIFO {fifo.name} has top-level-visible consumer connections "
                f"{externally_visible}"
            )
        removed_ports.update(f"{root}_{suffix}" for suffix in FIFO_PRODUCER_PORTS)
        declaration_sources = {
            "num_data_valid": (f"{root}_num_data_valid", "output"),
            "fifo_cap": (f"{root}_fifo_cap", "output"),
            "dout": (f"{root}_din", "output"),
            "empty_n": (f"{root}_full_n", "output"),
            "read": (f"{root}_write", "input"),
        }
        for suffix, fifo_port in {**FIFO_SHARED_PORTS, **FIFO_CONSUMER_PORTS}.items():
            wrapper_port = f"{root}_{suffix}"
            source_port, direction = declaration_sources[suffix]
            wrapper_decls[wrapper_port] = change_declaration_direction(
                replace_declared_name(
                    outer_decls[source_port], source_port, wrapper_port
                ),
                direction,
            )
            wrapper_connections[wrapper_port] = fifo.connections[fifo_port]
        owned.append(
            {
                "channel_id": bundle.get("channel_id"),
                "stream": bundle.get("stream"),
                "root": root,
                "fifo_module": fifo.module,
                "fifo_instance": fifo.name,
                "parent_module": manifest["top"],
                "fifo_connections": fifo.connections,
            }
        )

    for port in removed_ports:
        wrapper_decls.pop(port, None)
        wrapper_connections.pop(port, None)
    wrapper_ports = [port for port in outer_ports if port not in removed_ports]
    for bundle in outgoing:
        root = str(bundle["root"])
        for suffix in (*FIFO_SHARED_PORTS, *FIFO_CONSUMER_PORTS):
            port = f"{root}_{suffix}"
            if port not in wrapper_ports:
                wrapper_ports.append(port)

    lines = [f"module {wrapper_name} (", "  " + ",\n  ".join(wrapper_ports), ");"]
    lines.extend("  " + wrapper_decls[port] for port in wrapper_ports)
    for port in sorted(removed_ports):
        lines.append("  " + wire_declaration(outer_decls[port], port))
    lines.append(f"  {outer.name} producer_kernel (")
    lines.append(",\n".join(f"    .{port}({port})" for port in outer_ports))
    lines.append("  );")

    expression_to_pe_port = {
        normalize_expression(expression): port
        for port, expression in pe_instance.connections.items()
    }
    for item in owned:
        root = item["root"]
        fifo_connections = []
        for fifo_port, expression in item["fifo_connections"].items():
            producer_suffix = next(
                (suffix for suffix, name in FIFO_PRODUCER_PORTS.items() if name == fifo_port),
                None,
            )
            shared_suffix = next(
                (suffix for suffix, name in FIFO_SHARED_PORTS.items() if name == fifo_port),
                None,
            )
            consumer_suffix = next(
                (suffix for suffix, name in FIFO_CONSUMER_PORTS.items() if name == fifo_port),
                None,
            )
            if producer_suffix:
                connected = f"{root}_{producer_suffix}"
            elif shared_suffix:
                connected = f"{root}_{shared_suffix}"
            elif consumer_suffix:
                connected = f"{root}_{consumer_suffix}"
            else:
                connected = expression_to_pe_port.get(
                    normalize_expression(expression), expression
                )
            fifo_connections.append(f"    .{fifo_port}({connected})")
        lines.append(f"  {item['fifo_module']} folded_{item['fifo_instance']} (")
        lines.append(",\n".join(fifo_connections))
        lines.append("  );")
    lines.extend(["endmodule", ""])

    wrapper_text = "\n".join(lines)
    wrapper_block = module_blocks(wrapper_text)[wrapper_name]
    wrapper_decls = port_declarations(wrapper_block, port_names(wrapper_block))
    # Retain semantic side/direction but replace each folded producer bundle
    # with the consumer-facing FIFO boundary pins.
    for bundle in intent["stream_bundles"]:
        if bundle["direction"] == "out":
            root = str(bundle["root"])
            bundle["rtl_ports"] = [
                f"{root}_{suffix}"
                for suffix in (*FIFO_SHARED_PORTS, *FIFO_CONSUMER_PORTS)
            ]
        bundle["rtl_port_widths"] = {
            port: declaration_width(wrapper_decls[port])
            for port in bundle["rtl_ports"]
        }
    wrapper_port_set = set(port_names(wrapper_block))
    stream_ports = {
        port for bundle in intent["stream_bundles"] for port in bundle["rtl_ports"]
    }
    remaining = [port for port in wrapper_ports if port not in stream_ports]
    intent["module"] = wrapper_name
    intent["clock_pins"] = [port for port in remaining if port in {"ap_clk", "clk"}]
    intent["control_pins"] = [
        port for port in remaining if port.startswith("ap_") and port != "ap_clk"
    ]
    intent["auxiliary_pins"] = [
        port for port in remaining
        if port not in intent["clock_pins"] and port not in intent["control_pins"]
    ]
    stream_sides = {item["side"] for item in intent["stream_bundles"]}
    side_loads = {side: 0 for side in ("N", "S", "E", "W")}
    for bundle in intent["stream_bundles"]:
        side_loads[bundle["side"]] += sum(
            declaration_width(wrapper_decls[port]) for port in bundle["rtl_ports"]
        )
    intent["auxiliary_pin_sides"] = balance_auxiliary_pins(
        intent["auxiliary_pins"], wrapper_decls, stream_sides, side_loads
    )
    for port, side in intent["auxiliary_pin_sides"].items():
        side_loads[side] += declaration_width(wrapper_decls[port])
    control_candidates = [
        side for side in ("W", "N", "S", "E") if side not in stream_sides
    ] or ["W", "N", "S", "E"]
    intent["control_side"] = min(
        control_candidates,
        key=lambda side: (side_loads[side], control_candidates.index(side)),
    )
    intent["clock_side"] = intent["control_side"]
    if set(wrapper_decls) != wrapper_port_set:
        extra = set(wrapper_decls) - wrapper_port_set
        if extra - removed_ports:
            raise ValueError(f"wrapper declaration mismatch: {sorted(extra)}")
    return wrapper_text, {"owned_fifos": owned, "pin_intent": intent}, wrapper_connections


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
CATAPULT_STREAM_SUFFIXES = ("_rsc_dat", "_rsc_vld", "_rsc_rdy")


def stream_bundle_root(port: str) -> str | None:
    """Return a Vitis or Catapult interface-bundle root."""
    for suffix in CATAPULT_STREAM_SUFFIXES:
        if port.endswith(suffix):
            return port[: -len(suffix)]
    for suffix in STREAM_SUFFIXES:
        if port.endswith(suffix):
            return port[: -len(suffix)]
    return None


def stream_bundles(block: ModuleBlock) -> list[dict[str, object]]:
    """Find Vitis FIFO or Catapult ready/valid bundles in declaration order.

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
        catapult_triplet = {
            f"{root}_rsc_dat",
            f"{root}_rsc_vld",
            f"{root}_rsc_rdy",
        }
        if catapult_triplet.issubset(names):
            data_direction = (
                "in" if declarations[f"{root}_rsc_dat"].lstrip().startswith("input")
                else "out"
            )
            direction = data_direction
            protocol = "catapult_ready_valid"
        elif any(name.endswith(CATAPULT_STREAM_SUFFIXES) for name in names):
            # Direct-array Catapult arguments have ``*_rsc_dat`` without the
            # ready/valid pair. They are boundary auxiliary ports, not streams.
            continue
        elif f"{root}_dout" in names:
            direction = "in"
            protocol = "vitis_fifo"
        elif f"{root}_din" in names or f"{root}_ap_vld" in names:
            direction = "out"
            protocol = "vitis_fifo"
        else:
            continue
        # Retain only actual top-level ports and validate their declarations.
        bundles.append(
            {
                "root": root,
                "direction": direction,
                "protocol": protocol,
                "rtl_ports": [name for name in names if name in declarations],
            }
        )
    return bundles


def semantic_stream_width(port: dict[str, object]) -> int | None:
    """Return the flattened payload width of a manifest stream port."""
    type_text = str(port.get("type", ""))
    memref = re.search(r"memref<((?:\d+x)*)i(\d+)", type_text)
    if memref is not None:
        dimensions = [int(value) for value in re.findall(r"\d+", memref.group(1))]
        elements = 1
        for dimension in dimensions:
            elements *= dimension
        return elements * int(memref.group(2))
    scalar = re.search(r"(?:^|[<,])\s*i(\d+)\b", type_text)
    return int(scalar.group(1)) if scalar is not None else None


def rtl_stream_width(
    block: ModuleBlock, bundle: dict[str, object]
) -> int | None:
    """Return the data payload width of one generated Vitis stream bundle."""
    ports = port_names(block)
    declarations = port_declarations(block, ports)
    root = str(bundle["root"])
    candidates = (
        [f"{root}_rsc_dat"]
        if bundle.get("protocol") == "catapult_ready_valid"
        else (
            [f"{root}_dout"]
            if bundle["direction"] == "in"
            else [f"{root}_din", root]
        )
    )
    data_port = next((name for name in candidates if name in declarations), None)
    if data_port is None:
        return None
    return declaration_width(declarations[data_port])


def ordered_semantic_port_subset(
    block: ModuleBlock,
    semantic_ports: list[dict[str, object]],
    bundles: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    """Find a unique ordered semantic subset matching RTL direction and width.

    Vitis can move loop-invariant control streams into a parent wrapper while
    leaving a mixed-direction subset in a generated pipeline process. Matching
    both payload shape and order avoids guessing which semantic stream vanished.
    """
    bundle_widths = [rtl_stream_width(block, bundle) for bundle in bundles]
    semantic_widths = [semantic_stream_width(port) for port in semantic_ports]
    if any(width is None for width in bundle_widths + semantic_widths):
        return None

    solutions: list[list[dict[str, object]]] = []

    def search(bundle_index: int, semantic_index: int, selected: list[dict[str, object]]):
        if len(solutions) > 1:
            return
        if bundle_index == len(bundles):
            solutions.append(list(selected))
            return
        remaining = len(bundles) - bundle_index
        stop = len(semantic_ports) - remaining + 1
        for index in range(semantic_index, stop):
            semantic = semantic_ports[index]
            bundle = bundles[bundle_index]
            if (
                semantic.get("direction") == bundle["direction"]
                and semantic_widths[index] == bundle_widths[bundle_index]
            ):
                search(bundle_index + 1, index + 1, [*selected, semantic])

    search(0, 0, [])
    return solutions[0] if len(solutions) == 1 else None


def parent_wrapper_semantic_port_subset(
    pe: dict[str, object],
    canonical: ModuleBlock,
    semantic_ports: list[dict[str, object]],
    bundles: list[dict[str, object]],
    blocks: dict[str, ModuleBlock] | None,
) -> list[dict[str, object]] | None:
    """Map a split process through its complete parent-wrapper interface.

    Generated Vitis processes retain their parent's opaque stream-bundle roots,
    even when Vitis removes loop-invariant streams or reorders the remaining
    process ports.  First map the complete parent interface positionally, then
    use those stable roots to select and order the child process's semantics.
    """
    if blocks is None:
        return None
    matching_records = [
        record
        for record in pe.get("post_hls_records", [])
        if any(
            module.get("name") == canonical.name
            for module in record.get("rtl_modules", [])
        )
    ]
    if len(matching_records) != 1:
        return None
    parent_names = {
        Path(instance["parent_file"]).stem
        for instance in matching_records[0].get("rtl_instances", [])
        if instance.get("parent_file")
    }
    solutions: list[list[dict[str, object]]] = []
    for parent_name in parent_names:
        parent = blocks.get(parent_name)
        if parent is None:
            continue
        parent_bundles = stream_bundles(parent)
        if len(parent_bundles) != len(semantic_ports):
            continue
        compatible = all(
            semantic.get("direction") == bundle["direction"]
            and semantic_stream_width(semantic) == rtl_stream_width(parent, bundle)
            and semantic_stream_width(semantic) is not None
            for semantic, bundle in zip(semantic_ports, parent_bundles)
        )
        if not compatible:
            continue
        semantic_by_root = {
            str(bundle["root"]): semantic
            for semantic, bundle in zip(semantic_ports, parent_bundles)
        }
        if all(str(bundle["root"]) in semantic_by_root for bundle in bundles):
            solutions.append(
                [semantic_by_root[str(bundle["root"])] for bundle in bundles]
            )
    unique = {
        tuple(int(port["ordinal"]) for port in solution): solution
        for solution in solutions
    }
    return next(iter(unique.values())) if len(unique) == 1 else None


def axis_hint(stream: str) -> str | None:
    lowered = stream.lower()
    if any(token in lowered for token in ("horizontal", "hori", "east", "west")):
        return "horizontal"
    if any(token in lowered for token in ("vertical", "vert", "north", "south")):
        return "vertical"
    return None


def semantic_stream_pin_weight(port: dict) -> int:
    """Estimate the physical pin count of one FIFO-style stream interface."""
    for key in ("width", "bitwidth", "dtype_bits"):
        value = port.get(key)
        if isinstance(value, int) and value > 0:
            return value + 2
    match = re.search(
        r"(?:^|[<,\s])i(\d+)(?:[,>\s]|$)", str(port.get("type", ""))
    )
    return (int(match.group(1)) if match is not None else 1) + 2


def neighboring_pid_side(here: list, there: list) -> str | None:
    """Return the physical side for an immediately adjacent logical PE."""
    if not here or len(here) != len(there):
        return None
    differences = [int(other) - int(current) for current, other in zip(here, there)]
    changed = [index for index, difference in enumerate(differences) if difference]
    if len(changed) != 1 or abs(differences[changed[0]]) != 1:
        return None
    axis = changed[0]
    difference = differences[axis]
    if axis == len(differences) - 1:
        return "E" if difference > 0 else "W"
    if axis == len(differences) - 2:
        return "S" if difference > 0 else "N"
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

    Explicit directions win and immediate same-kernel neighbors use their real
    relative direction. All other traffic is assigned as complete, weighted
    bundles to the least-loaded macro side. The balancing key depends only on
    local interface structure, preserving physical-macro reuse.
    """
    pes = {item["semantic_id"]: item for item in manifest.get("pe_instances", [])}
    result: dict[tuple[str, int], dict[str, str]] = {}
    side_loads = {
        pe_id: {side: 0 for side in ("N", "S", "E", "W")} for pe_id in pes
    }
    deferred: dict[str, list[tuple[int, int, str]]] = {pe_id: [] for pe_id in pes}
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
                weight = semantic_stream_pin_weight(port)
                if explicit in {"N", "S", "E", "W"}:
                    side, method = explicit, "manifest_explicit"
                elif peer is not None and pes[peer["pe"]].get("kernel") == pe.get("kernel"):
                    here = list(pe.get("pid", []))
                    there = list(pes[peer["pe"]].get("pid", []))
                    side = neighboring_pid_side(here, there)
                    if side is not None:
                        method = "same_kernel_neighbor"
                    else:
                        deferred[pe_id].append((weight, ordinal, str(port.get("direction", ""))))
                        continue
                else:
                    deferred[pe_id].append((weight, ordinal, str(port.get("direction", ""))))
                    continue
                result[(pe_id, ordinal)] = {"side": side, "method": method}
                side_loads[pe_id][side] += weight

    base_order = ("W", "N", "S", "E")
    for pe_id, pending in deferred.items():
        if not pending:
            continue
        signature = ";".join(
            f"{ordinal}:{direction}:{weight}"
            for weight, ordinal, direction in sorted(pending, key=lambda item: item[1])
        )
        rotation = int(hashlib.sha256(signature.encode()).hexdigest()[:8], 16) % 4
        side_order = base_order[rotation:] + base_order[:rotation]
        for weight, ordinal, _direction in sorted(
            pending, key=lambda item: (-item[0], item[1], item[2])
        ):
            side = min(
                side_order,
                key=lambda candidate: (
                    side_loads[pe_id][candidate], side_order.index(candidate)
                ),
            )
            result[(pe_id, ordinal)] = {
                "side": side,
                "method": "non_neighbor_load_balance",
            }
            side_loads[pe_id][side] += weight
    return result


def build_pin_intent(
    manifest: dict,
    representative: str,
    canonical: ModuleBlock,
    blocks: dict[str, ModuleBlock] | None = None,
) -> dict[str, object]:
    pes = {item["semantic_id"]: item for item in manifest.get("pe_instances", [])}
    if representative not in pes:
        raise ValueError(f"macro representative {representative} is absent from pe_instances")
    pe = pes[representative]
    semantic_ports = sorted(pe.get("ports", []), key=lambda item: int(item["ordinal"]))
    bundles = stream_bundles(canonical)
    selection_method = "complete_process_interface"
    if len(bundles) != len(semantic_ports):
        selected_subset = parent_wrapper_semantic_port_subset(
            pe, canonical, semantic_ports, bundles, blocks
        )
        if selected_subset is not None:
            semantic_ports = selected_subset
            selection_method = "parent_wrapper_bundle_identity"
        else:
            selected_subset = ordered_semantic_port_subset(
                canonical, semantic_ports, bundles
            )
        if selected_subset is not None:
            semantic_ports = selected_subset
            if selection_method == "complete_process_interface":
                selection_method = "ordered_direction_width_subset"
        else:
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
    for bundle in mapped:
        bundle["rtl_port_widths"] = {
            port: declaration_width(declarations[port])
            for port in bundle["rtl_ports"]
        }
    remaining = [port for port in canonical_ports if port not in stream_rtl_ports]
    clock = [port for port in remaining if port in {"ap_clk", "clk"}]
    control = [
        port
        for port in remaining
        if (port.startswith("ap_") and port != "ap_clk") or port == "rst"
    ]
    auxiliary = [
        port for port in remaining if port not in control and port not in clock
    ]
    stream_sides = {item["side"] for item in mapped}
    side_loads = {side: 0 for side in ("N", "S", "E", "W")}
    for bundle in mapped:
        side_loads[bundle["side"]] += sum(
            declaration_width(declarations[port]) for port in bundle["rtl_ports"]
        )
    auxiliary_pin_sides = balance_auxiliary_pins(
        auxiliary, declarations, stream_sides, side_loads
    )
    for port, side in auxiliary_pin_sides.items():
        side_loads[side] += declaration_width(declarations[port])
    control_candidates = [
        side for side in ("W", "N", "S", "E") if side not in stream_sides
    ] or ["W", "N", "S", "E"]
    control_side = min(
        control_candidates,
        key=lambda side: (side_loads[side], control_candidates.index(side)),
    )
    return {
        "schema_version": 1,
        "representative": representative,
        "module": canonical.name,
        "stream_bundles": mapped,
        "clock_pins": clock,
        "clock_side": control_side,
        "control_pins": control,
        "control_side": control_side,
        "auxiliary_pins": auxiliary,
        "auxiliary_pin_sides": auxiliary_pin_sides,
        "semantic_to_rtl_method": selection_method,
    }


def write_pin_intent_tcl(path: Path, intent: dict[str, object]) -> None:
    side_ports = {side: [] for side in ("N", "S", "E", "W")}
    side_groups = {side: [] for side in ("N", "S", "E", "W")}
    split_bundles = []
    for bundle in intent["stream_bundles"]:
        if bundle.get("method") == "non_neighbor_load_balance":
            widths = bundle.get("rtl_port_widths", {})
            data_port = max(
                bundle["rtl_ports"],
                key=lambda port: (int(widths.get(port, 1)), port),
            )
            split_bundles.append((
                data_port,
                int(widths.get(data_port, 1)),
                [port for port in bundle["rtl_ports"] if port != data_port],
            ))
            continue
        side_ports[bundle["side"]].extend(bundle["rtl_ports"])
        side_groups[bundle["side"]].append(list(bundle["rtl_ports"]))
    side_ports[intent["control_side"]].extend(intent["control_pins"])
    if intent["control_pins"]:
        side_groups[intent["control_side"]].append(list(intent["control_pins"]))
    auxiliary_groups = {}
    for port in intent["auxiliary_pins"]:
        side = intent["auxiliary_pin_sides"][port]
        side_ports[side].append(port)
        auxiliary_groups.setdefault((side, auxiliary_group(port)), []).append(port)
    for (side, _name), ports in auxiliary_groups.items():
        side_groups[side].append(ports)
    lines = ["# Generated from the Allo whole-region channel graph."]
    lines.append("set allo_asic_non_neighbor_split_bundles [list]")
    for data_port, data_width, handshake_ports in split_bundles:
        handshakes = " ".join(tcl_quote(port) for port in handshake_ports)
        lines.append(
            "lappend allo_asic_non_neighbor_split_bundles "
            f"[list {tcl_quote(data_port)} {data_width} [list {handshakes}]]"
        )
    lines.append(
        "set allo_asic_clock_pins [list "
        + " ".join(tcl_quote(port) for port in intent["clock_pins"])
        + "]"
    )
    lines.append(f"set allo_asic_clock_side {intent['clock_side']}")
    for side in ("N", "S", "E", "W"):
        lines.append(
            f"set allo_asic_signal_pins_{side} [list "
            + " ".join(tcl_quote(port) for port in side_ports[side])
            + "]"
        )
        groups = " ".join(
            "[list " + " ".join(tcl_quote(port) for port in group) + "]"
            for group in side_groups[side]
        )
        lines.append(f"set allo_asic_signal_pin_groups_{side} [list {groups}]")
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
    # Non-neighbor bundles are divided across multiple sides later, after the
    # macro floorplan exposes real edge capacities. Their provisional planner
    # side is therefore not a physical orientation constraint. Only explicit
    # and neighbor-facing bundles can require rotating/mirroring a reused
    # macro instance.
    return {
        port: bundle["side"]
        for bundle in intent["stream_bundles"]
        if bundle.get("method") != "non_neighbor_load_balance"
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


def write_constraints(path: Path, clock_period: float, clock_pin: str) -> None:
    path.write_text(
        "# Generic HLS macro constraints generated by allo-asic-macro-plan\n"
        f"create_clock -name clk -period {clock_period:g} [get_ports {clock_pin}]\n"
        "set nonclock_inputs "
        f"[remove_from_collection [all_inputs] [get_ports {clock_pin}]]\n"
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


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def classify_macro_candidate(members: list[dict]) -> str:
    """Distinguish semantic PE classes from repeated children of one PE."""
    semantic_ids = [str(member.get("semantic_id", "")) for member in members]
    return (
        "repeated_hls_submodule"
        if len(set(semantic_ids)) < len(semantic_ids)
        else "semantic_pe"
    )


def wrapper_interface_signature(wrapper_text: str, wrapper_name: str, folding: dict) -> str:
    block = module_blocks(wrapper_text)[wrapper_name]
    ports = port_names(block)
    declarations = port_declarations(block, ports)
    shapes = []
    for port in ports:
        shapes.append(
            normalized_declaration(
                declarations[port].replace(port, "PORT")
            )
        )
    payload = {
        "port_shapes": shapes,
        "fifo_modules": [
            item["fifo_module"] for item in folding.get("owned_fifos", [])
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
    backend = os.environ.get("backend", manifest.get("backend", "vitis"))
    blocks = module_blocks(rtl)
    threshold = int(os.environ.get("min_macro_reuse", "2"))
    macro_clock_period = float(os.environ.get("macro_clock_period", "8.0"))
    bypassed = env_bool("bypass_macro_generation")
    harden_hls_submodules = env_bool("harden_repeated_hls_submodules")
    fold_fifos_requested = env_bool("fold_fifos_into_macro")
    fold_fifos = fold_fifos_requested and not bypassed and backend == "vitis"
    if threshold < 1:
        raise ValueError("min_macro_reuse must be at least 1")
    if macro_clock_period <= 0:
        raise ValueError("macro_clock_period must be positive")
    if backend not in {"vitis", "catapult"}:
        raise ValueError(f"unsupported macro-plan backend {backend!r}")
    manifest_backend = manifest.get("backend")
    if manifest_backend and manifest_backend != backend:
        raise ValueError(
            f"macro-plan backend {backend!r} does not match manifest "
            f"backend {manifest_backend!r}"
        )

    summary = manifest.get("summary", {})
    if not bypassed and (
        summary.get("unmatched_or_ambiguous", 0)
        or summary.get("unjoined_post_hls_records", 0)
    ):
        raise ValueError("final ASIC manifest contains unmatched or unjoined records")

    pes = {item["semantic_id"]: item for item in manifest.get("pe_instances", [])}
    top_block = blocks.get(str(manifest.get("top")))
    if fold_fifos and top_block is None:
        raise ValueError("FIFO folding requires the realized top RTL module")
    top_instances = module_instances(top_block, set(blocks)) if top_block else []

    planning_groups = []
    for group in manifest.get("macro_groups", []):
        if not fold_fifos:
            planning_groups.append(group)
            continue
        original_members = group.get("members", [])
        if group.get("proof", {}).get("status") != "proven":
            continue
        if int(group.get("member_count", len(original_members))) < threshold:
            continue
        if classify_macro_candidate(original_members) != "semantic_pe":
            continue
        members_by_signature = {}
        for member in original_members:
            pe = pes[member["semantic_id"]]
            outer_name = outer_module_for_member(pe, member["rtl_module"])
            instances = [item for item in top_instances if item.module == outer_name]
            if len(instances) != 1:
                raise ValueError(
                    f"expected one top-level instance of {outer_name}, found "
                    f"{[item.name for item in instances]}"
                )
            provisional_name = f"{outer_name}_fifo_signature"
            wrapper_text, folding, _connections = generate_fifo_wrapper(
                provisional_name,
                pe,
                blocks[outer_name],
                instances[0],
                top_instances,
                manifest,
                blocks,
            )
            signature = wrapper_interface_signature(
                wrapper_text, provisional_name, folding
            )
            members_by_signature.setdefault(signature, []).append(member)
        split = len(members_by_signature) > 1
        for signature, variant_members in sorted(members_by_signature.items()):
            class_id = group["macro_class_id"]
            if split:
                class_id = f"{class_id}_fifo_{signature[:8]}"
            planning_groups.append(
                {
                    **group,
                    "macro_class_id": class_id,
                    "source_macro_class_id": group["macro_class_id"],
                    "representative": variant_members[0]["semantic_id"],
                    "member_count": len(variant_members),
                    "members": variant_members,
                    "fifo_wrapper_signature": signature,
                }
            )

    selected = []
    replacements: dict[str, str] = {}
    removed: set[str] = set()
    for group in ([] if bypassed else planning_groups):
        members = group.get("members", [])
        if group.get("proof", {}).get("status") != "proven":
            continue
        if int(group.get("member_count", len(members))) < threshold:
            continue
        semantic_ids = [str(member.get("semantic_id", "")) for member in members]
        candidate_kind = classify_macro_candidate(members)
        if candidate_kind == "repeated_hls_submodule" and not harden_hls_submodules:
            continue
        owning_kernels = sorted(
            {
                "/".join(value.split("/")[:2])
                for value in semantic_ids
                if len(value.split("/")) >= 2
            }
        )
        module_names = [member["rtl_module"] for member in members]
        missing = [name for name in module_names if name not in blocks]
        if missing:
            raise ValueError(
                f"class {group['macro_class_id']} references missing RTL modules {missing}"
            )
        entry_id = group["macro_class_id"]
        folding_by_module = {}
        wrapper_text_by_module = {}
        prepared_members = []
        if fold_fifos:
            if candidate_kind != "semantic_pe":
                continue
            for index, member in enumerate(members):
                semantic_id = member["semantic_id"]
                pe = pes[semantic_id]
                inner_module = member["rtl_module"]
                outer_name = outer_module_for_member(pe, inner_module)
                if outer_name not in blocks:
                    raise ValueError(f"outer kernel module is absent: {outer_name}")
                instances = [
                    item for item in top_instances if item.module == outer_name
                ]
                if len(instances) != 1:
                    raise ValueError(
                        f"expected one top-level instance of {outer_name}, found "
                        f"{[item.name for item in instances]}"
                    )
                wrapper_name = (
                    f"allo_fifo_{entry_id}"
                    if index == 0
                    else f"{outer_name}_fifo_member"
                )
                wrapper_text, folding, wrapper_connections = generate_fifo_wrapper(
                    wrapper_name,
                    pe,
                    blocks[outer_name],
                    instances[0],
                    top_instances,
                    manifest,
                    blocks,
                )
                folding.update(
                    {
                        "enabled": True,
                        "inner_rtl_module": inner_module,
                        "source_module": outer_name,
                        "source_instance": instances[0].name,
                        "semantic_instance_name": semantic_instance_name(pe),
                        "wrapper_module": wrapper_name,
                        "wrapper_connections": wrapper_connections,
                    }
                )
                prepared = {
                    **member,
                    "inner_rtl_module": inner_module,
                    "rtl_module": outer_name,
                    "source_instance": instances[0].name,
                    "semantic_instance_name": semantic_instance_name(pe),
                    "fifo_folding": folding,
                }
                prepared_members.append(prepared)
                folding_by_module[outer_name] = folding
                wrapper_text_by_module[outer_name] = wrapper_text
            module_names = [member["rtl_module"] for member in prepared_members]
            members = prepared_members

        canonical_name = module_names[0]
        canonical_wrapper_name = f"allo_fifo_{entry_id}" if fold_fifos else canonical_name
        canonical_text = (
            wrapper_text_by_module[canonical_name]
            if fold_fifos else blocks[canonical_name].text
        )
        canonical = module_blocks(canonical_text)[canonical_wrapper_name]
        entry_dir = batch_dir / "entries" / entry_id
        rtl_dir = entry_dir / "rtl"
        rtl_dir.mkdir(parents=True)

        port_maps = {}
        for member_name in module_names:
            member_block = (
                module_blocks(wrapper_text_by_module[member_name])[
                    folding_by_module[member_name]["wrapper_module"]
                ]
                if fold_fifos else blocks[member_name]
            )
            mapping = validate_and_map_ports(canonical, member_block)
            port_maps[member_name] = mapping
            removed.add(member_name)
            if member_name != canonical_name and not fold_fifos:
                replacements[member_name] = alias_wrapper(
                    canonical, blocks[member_name], mapping
                )

        if fold_fifos:
            design_blocks = dict(blocks)
            design_blocks[canonical_wrapper_name] = canonical
            dependencies = dependency_closure(canonical_wrapper_name, design_blocks)
            design_text = "\n".join(
                canonical_text if name == canonical_wrapper_name else blocks[name].text
                for name in dependencies
            )
        else:
            dependencies = dependency_closure(canonical_name, blocks)
            design_text = "\n".join(blocks[name].text for name in dependencies)
        design_file = rtl_dir / "design.v"
        design_file.write_text(design_text.rstrip() + "\n")
        pin_intent = (
            folding_by_module[canonical_name]["pin_intent"]
            if fold_fifos else
            build_pin_intent(manifest, group["representative"], canonical, blocks)
        )
        clock_pins = pin_intent["clock_pins"]
        if len(clock_pins) != 1:
            raise ValueError(
                f"macro {canonical_wrapper_name} must have exactly one recognized "
                f"clock port (ap_clk or clk), found {clock_pins}"
            )
        constraints_file = entry_dir / "constraints.tcl"
        write_constraints(
            constraints_file, macro_clock_period, str(clock_pins[0])
        )
        pin_intent_json = entry_dir / "pin-intent.json"
        pin_intent_tcl = entry_dir / "pin-intent.tcl"
        pin_intent_json.write_text(json.dumps(pin_intent, indent=2) + "\n")
        write_pin_intent_tcl(pin_intent_tcl, pin_intent)
        member_placements = []
        for member in members:
            member_module = member["rtl_module"]
            member_intent = (
                folding_by_module[member_module]["pin_intent"]
                if fold_fifos else
                build_pin_intent(
                    manifest, member["semantic_id"], blocks[member_module], blocks
                )
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
        proof = group.get("proof", {})
        rtl_audit = group.get("rtl_audit", {})
        rtl_audit_hashes = sorted(rtl_audit.get("distinct_hashes", []))
        representative_records = pes[group["representative"]].get(
            "post_hls_records", []
        )
        representative_rtl_hash = next(
            (
                record.get("rtl_equivalence_hash")
                for record in representative_records
                if record.get("rtl_equivalence_hash")
            ),
            None,
        )
        compatibility_rtl_hash = (
            representative_rtl_hash
            or (rtl_audit_hashes[0] if len(rtl_audit_hashes) == 1 else None)
            or proof.get("rtl_hash")
        )
        if compatibility_rtl_hash is None:
            raise ValueError(
                f"class {group['macro_class_id']} has no representative RTL hash"
            )
        entry = {
            "id": entry_id,
            "source_macro_class_id": group.get(
                "source_macro_class_id", group["macro_class_id"]
            ),
            "top_module": canonical_wrapper_name,
            "representative_semantic_id": group["representative"],
            "reuse_count": int(group["member_count"]),
            "candidate_kind": candidate_kind,
            "owning_kernels": owning_kernels,
            "implementation_contract_hash": proof.get(
                "implementation_contract_hash", proof.get("rtl_hash")
            ),
            "equivalence_method": proof.get("method", "rtl_hash"),
            "rtl_audit_status": rtl_audit.get("status", "legacy_authoritative"),
            "rtl_audit_hashes": rtl_audit_hashes or [compatibility_rtl_hash],
            # Retained for downstream compatibility. Equivalence authority is
            # carried by implementation_contract_hash/equivalence_method.
            "rtl_hash": compatibility_rtl_hash,
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
            "fold_fifos_into_macro": fold_fifos,
            "folded_fifo_count": sum(
                len(item.get("fifo_folding", {}).get("owned_fifos", []))
                for item in members
            ),
        }
        (entry_dir / "entry.json").write_text(json.dumps(entry, indent=2) + "\n")
        selected.append(entry)

    if not selected and not bypassed:
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
        "backend": backend,
        "reuse_threshold": threshold,
        "macro_clock_period_ns": macro_clock_period,
        "bypass_macro_generation": bypassed,
        "harden_repeated_hls_submodules": harden_hls_submodules,
        "fold_fifos_into_macro": fold_fifos,
        "fold_fifos_requested": fold_fifos_requested,
        "implementation_style": "flat" if bypassed else "hierarchical_macros",
        "source_manifest": str(manifest_path),
        "selected_class_count": len(selected),
        "selected_instance_count": sum(entry["reuse_count"] for entry in selected),
        "fifo_folding": {
            "enabled": fold_fifos,
            "ownership": "producer_put_side" if fold_fifos else "disabled",
            "wrapped_pe_instance_count": sum(
                entry["reuse_count"] for entry in selected
                if entry.get("fold_fifos_into_macro")
            ),
            "folded_fifo_instance_count": sum(
                entry.get("folded_fifo_count", 0) for entry in selected
            ),
        },
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
        f"set allo_asic_bypass_macro_generation {1 if bypassed else 0}",
        f"set allo_asic_harden_repeated_hls_submodules {1 if harden_hls_submodules else 0}",
        f"set allo_asic_fold_fifos_into_macro {1 if fold_fifos else 0}",
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
                f"set allo_asic_macro_candidate_kind({key}) {tcl_quote(entry['candidate_kind'])}",
                f"set allo_asic_macro_rtl({key}) {tcl_quote(entry['rtl'])}",
            ]
        )
    (outputs / "macro-plan.tcl").write_text("\n".join(tcl_lines) + "\n")
    if bypassed:
        log = "Macro generation bypassed; normalized RTL remains entirely standard-cell logic.\n"
    else:
        log = (
            f"Selected {len(selected)} proven macro classes covering "
            f"{plan['selected_instance_count']} instances at reuse threshold {threshold}"
            ".\n"
        )
        if backend == "catapult" and fold_fifos_requested:
            log += (
                "Catapult FIFO folding request ignored; ready/valid support logic "
                "remains in each macro dependency closure.\n"
            )
        if fold_fifos:
            log += (
                "Producer-owned FIFO folding wrapped "
                f"{plan['fifo_folding']['wrapped_pe_instance_count']} PE instances "
                f"and internalized {plan['fifo_folding']['folded_fifo_instance_count']} "
                "FIFO instances.\n"
            )
    (outputs / "macro-plan.log").write_text(log)
    print(log, end="")


if __name__ == "__main__":
    main()
