#!/usr/bin/env python3
"""Validate Stage-1 products and plan full-chip macro substitution."""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path


IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
MODULE_RE = re.compile(rf"(?<![A-Za-z0-9_$])module\s+(?P<name>{IDENT})\b")
ENDMODULE_RE = re.compile(r"(?<![A-Za-z0-9_$])endmodule\b")
INSTANCE_RE_TEMPLATE = r"(?m)(?<![A-Za-z0-9_$]){module}\s+(?P<instance>{ident})\s*\("
REQUIRED_VIEWS = {"verilog", "liberty", "db", "lef", "gds"}


def module_blocks(text: str) -> dict[str, tuple[int, int, str]]:
    blocks = {}
    cursor = 0
    while match := MODULE_RE.search(text, cursor):
        end = ENDMODULE_RE.search(text, match.end())
        if end is None:
            raise ValueError(f"module {match.group('name')} has no endmodule")
        name = match.group("name")
        if name in blocks:
            raise ValueError(f"duplicate module definition: {name}")
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start]
        if re.fullmatch(r"\s*(?:\(\*.*?\*\)\s*)*", prefix, re.DOTALL):
            start = line_start
        blocks[name] = (start, end.end(), text[start : end.end()])
        cursor = end.end()
    return blocks


def find_instances(blocks: dict, module_name: str) -> list[dict]:
    pattern = re.compile(
        INSTANCE_RE_TEMPLATE.format(module=re.escape(module_name), ident=IDENT)
    )
    found = []
    for parent, (_start, _end, body) in blocks.items():
        if parent == module_name:
            continue
        for match in pattern.finditer(body):
            prefix = body[max(0, match.start() - 12) : match.start()]
            if re.search(r"\bmodule\s*$", prefix):
                continue
            found.append({"parent_module": parent, "instance_name": match.group("instance")})
    return found


def elaborated_instance_paths(blocks: dict, top: str) -> dict[tuple[str, str, str], list[str]]:
    """Expand syntactic module instances into top-rooted RTL hierarchy paths."""
    children: dict[str, list[tuple[str, str]]] = {name: [] for name in blocks}
    for child in blocks:
        for instance in find_instances(blocks, child):
            children[instance["parent_module"]].append((instance["instance_name"], child))
    result: dict[tuple[str, str, str], list[str]] = {}

    def visit(module: str, path: str, ancestry: tuple[str, ...]) -> None:
        if module in ancestry:
            raise ValueError(f"recursive RTL hierarchy through module {module}")
        for instance_name, child in children.get(module, []):
            child_path = f"{path}/{instance_name}"
            result.setdefault((module, instance_name, child), []).append(child_path)
            visit(child, child_path, ancestry + (module,))

    visit(top, top, ())
    return result


def tcl_quote(value: object) -> str:
    return "{" + str(value).replace("\\", "\\\\").replace("}", "\\}") + "}"


def stable_instance_name(member: dict, semantic_id: str, source_module: str) -> str:
    semantic_name = member.get("semantic_instance_name")
    if semantic_name:
        return str(semantic_name)
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", semantic_id).strip("_")
    digest = hashlib.sha256(f"{semantic_id}|{source_module}".encode()).hexdigest()[:10]
    return f"allo_pe_{slug}_{digest}"


def main() -> None:
    inputs = Path("inputs")
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    rtl = (inputs / "design.v").read_text()
    blocks = module_blocks(rtl)
    manifest = json.loads((inputs / "asic-manifest-final.json").read_text())
    registry = json.loads((inputs / "macro-registry.json").read_text())
    expected_top = os.environ.get("top_module", "top")
    if manifest.get("top") != expected_top or expected_top not in blocks:
        raise ValueError(
            f"top mismatch: requested={expected_top}, manifest={manifest.get('top')}, "
            f"RTL definition present={expected_top in blocks}"
        )
    hierarchy_paths = elaborated_instance_paths(blocks, expected_top)
    manifest_groups = {
        item["macro_class_id"]: item for item in manifest.get("macro_groups", [])
    }
    replacements = []
    classes = []
    replaced_modules = set()
    for macro in registry.get("macros", []):
        class_id = macro["macro_class_id"]
        group = manifest_groups.get(macro.get("source_macro_class_id", class_id))
        if group is None:
            raise ValueError(f"published class absent from final manifest: {class_id}")
        members = macro.get("members", group.get("members", []))
        if macro.get("reuse_count") != len(members):
            raise ValueError(f"reuse/member mismatch for {class_id}")
        canonical = macro["top_module"]
        folding_enabled = bool(macro.get("fold_fifos_into_macro", False))
        if canonical not in blocks and not folding_enabled:
            raise ValueError(f"canonical module is absent from normalized RTL: {canonical}")
        missing_views = REQUIRED_VIEWS - set(macro.get("views", {}))
        if missing_views:
            raise ValueError(f"{class_id} lacks required views: {sorted(missing_views)}")
        for view, detail in macro["views"].items():
            path = inputs / "macro-registry" / detail["path"]
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing published {view} view for {class_id}: {path}")
        member_records = []
        placement_by_module = {
            item.get("rtl_module"): item
            for item in macro.get("member_placements", [])
        }
        for member in members:
            source_module = member["rtl_module"]
            if source_module in replaced_modules:
                raise ValueError(f"RTL module assigned to multiple classes: {source_module}")
            replaced_modules.add(source_module)
            if source_module not in blocks:
                raise ValueError(f"selected member module absent from RTL: {source_module}")
            instances = find_instances(blocks, source_module)
            requested_instance = member.get("source_instance")
            if requested_instance:
                instances = [
                    item for item in instances
                    if item["instance_name"] == requested_instance
                ]
            if not instances:
                raise ValueError(f"selected member module is never instantiated: {source_module}")
            raw_map = macro.get("port_maps", {}).get(source_module)
            if not raw_map:
                raise ValueError(f"missing canonical port map for {source_module}")
            member_to_canonical = {item["member"]: item["canonical"] for item in raw_map}
            if len(member_to_canonical) != len(raw_map):
                raise ValueError(f"duplicate member port in map for {source_module}")
            for instance in instances:
                paths = hierarchy_paths.get(
                    (instance["parent_module"], instance["instance_name"], source_module),
                    [],
                )
                if not paths:
                    raise ValueError(
                        f"selected instance is unreachable from {expected_top}: "
                        f"{instance['parent_module']}/{instance['instance_name']}"
                    )
                replacement = {
                    **instance,
                    "semantic_id": member["semantic_id"],
                    "macro_class_id": class_id,
                    "candidate_kind": macro.get("candidate_kind", "semantic_pe"),
                    "owning_kernels": macro.get("owning_kernels", []),
                    "source_module": source_module,
                    "canonical_module": canonical,
                    "member_to_canonical_ports": member_to_canonical,
                    "desired_orientation": placement_by_module.get(
                        source_module, member
                    ).get("orientation", "unassigned"),
                    "stable_instance_name": stable_instance_name(
                        member, member["semantic_id"], source_module
                    ),
                    "hierarchical_paths": paths,
                }
                folding = member.get("fifo_folding", {})
                if folding.get("enabled"):
                    member_connections = folding.get("wrapper_connections", {})
                    replacement["explicit_canonical_connections"] = {
                        item["canonical"]: member_connections[item["member"]]
                        for item in raw_map
                    }
                    replacement["folded_fifos"] = folding.get("owned_fifos", [])
                replacements.append(replacement)
                member_records.append(replacement)
        classes.append(
            {
                "macro_class_id": class_id,
                "canonical_module": canonical,
                "reuse_count": macro["reuse_count"],
                "candidate_kind": macro.get("candidate_kind", "semantic_pe"),
                "owning_kernels": macro.get("owning_kernels", []),
                "lef_symmetry": macro.get("lef_symmetry", []),
                "members": member_records,
                "views": macro["views"],
                "fold_fifos_into_macro": folding_enabled,
            }
        )

    pe_ids = {item["semantic_id"] for item in manifest.get("pe_instances", [])}
    connections = []
    for channel in manifest.get("channels", []):
        endpoints = [item for item in channel.get("endpoints", []) if item.get("pe") in pe_ids]
        if endpoints:
            connections.append(
                {
                    "channel_id": channel.get("channel_id"),
                    "stream": channel.get("stream"),
                    "type": channel.get("type"),
                    "endpoints": endpoints,
                    "cross_kernel": len({
                        next(
                            pe.get("kernel")
                            for pe in manifest.get("pe_instances", [])
                            if pe["semantic_id"] == endpoint["pe"]
                        )
                        for endpoint in endpoints
                    }) > 1,
                }
            )
    plan = {
        "schema_version": 1,
        "stage": "full_chip_assembly_plan",
        "top_module": expected_top,
        "bypass_macro_generation": bool(
            registry.get("bypass_macro_generation", False)
        ),
        "implementation_style": registry.get(
            "implementation_style", "hierarchical_macros"
        ),
        "macro_class_count": len(classes),
        "replacement_instance_count": len(replacements),
        "elaborated_macro_instance_count": sum(
            len(item["hierarchical_paths"]) for item in replacements
        ),
        "replaced_module_count": len(replaced_modules),
        "folded_fifo_count": sum(
            len(item.get("folded_fifos", [])) for item in replacements
        ),
        "classes": classes,
        "replacements": replacements,
        "whole_region_connections": connections,
        "physical_intent_status": "unassigned; consumed by later physical-intent planning",
    }
    (outputs / "assembly-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    lines = [
        "set allo_asic_assembly_plan_schema_version 1",
        f"set allo_asic_assembly_top {tcl_quote(expected_top)}",
        "set allo_asic_replaced_modules [list "
        + " ".join(tcl_quote(name) for name in sorted(replaced_modules))
        + "]",
    ]
    for item in replacements:
        key = item["source_module"]
        lines.append(
            f"set allo_asic_replacement_canonical({key}) "
            f"{tcl_quote(item['canonical_module'])}"
        )
        lines.append(
            f"set allo_asic_replacement_class({key}) {tcl_quote(item['macro_class_id'])}"
        )
    (outputs / "assembly-plan.tcl").write_text("\n".join(lines) + "\n")
    report = [
        "Full-chip assembly plan",
        f"Top module: {expected_top}",
        f"Hardened macro classes: {len(classes)}",
        f"Replaced RTL module definitions: {len(replaced_modules)}",
        f"Replaced instances: {len(replacements)}",
        "Elaborated macro instances: "
        + str(sum(len(item["hierarchical_paths"]) for item in replacements)),
        f"Whole-region channels retained: {len(connections)}",
        "Physical orientations: intentionally unassigned until physical-intent planning",
    ]
    (outputs / "assembly-plan-report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
