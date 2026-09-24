# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``asic_manifest`` that ``allo/backend/asic`` consumes.

``allo/backend/asic`` -- Julian Bushlow's mflowgen flow, vendored here (see its
``PROVENANCE.md``) -- has two entry points. The *flat flow* takes a directory of
Verilog and a file list, and knows nothing about Allo. **AAAH**, the ``allo-*``
nodes, instead loads an Allo design and calls ``build(project, target, mode,
configs)`` with ``configs["asic_manifest"] = {...}``, asking the compiler to
emit the architecture it just compiled: unit boundaries, semantic equivalence
classes, stream bundles with widths and directions, PE grid coordinates. Its
planners then place macros on a grid from PID deltas and weight the edges by
pre-HLS stream bit width -- none of which survives in emitted Verilog.

This module is that emitter. Everything here is **derived from the realized
dataflow IR**, never from a table someone typed: a hand-written manifest would
be a second editing surface for the architecture, which is exactly what having
one Allo specification is meant to remove. The kernel instances, their grid
coordinates, their stream ports and the channels between them are read back out
of the module that Allo is about to hand to HLS, so the manifest cannot drift
from what is built without the build changing too.

Two files, two stages, matching what ``nodes/allo-asic-compilation`` validates:

``asic-manifest.json`` (``stage: "pre_hls"``)
    The architecture. Written at ``build()`` time, before any HLS tool runs, so
    it exists whether or not a licence does.

``asic-manifest-final.json`` (``stage: "post_hls_enriched"``)
    The same records joined to the RTL the backend produced -- each PE's root
    module and its split-process children, an equivalence hash per PE, and the
    macro classes those hashes induce. Written after synthesis succeeds.

Each has a ``.tcl`` sibling, because the flow's nodes stage both.

What this module does NOT do: the per-argument *Catapult* RTL protocol capture
(``catapult_argument``, ``data_ports``/``triosy_ports``, ``packing``,
``interface.roles``) that ``allo-testbench-generation`` wants on the
``catapult``/``systemc`` path. That is a Catapult-RTL parse, not an
architectural fact, and it is a separate script in the flow's own lineage.
The Vitis path -- the one TinyTPU's ASIC numbers came from -- is complete here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

SCHEMA_VERSION = 2

# `analyze_arg_load_store` vocabulary -> the manifest's `direction` vocabulary.
DIRECTION_MAP = {
    "in": "input",
    "out": "output",
    "both": "inout",
    "scalar": "input",
}

# The ops a `@df.region()` uses to declare a link. `allo.stream_construct` is
# the Stream; wire/channel are the SystemC emitter's additions.
_LINK_OPS = {
    "allo.stream_construct",
    "allo.wire_construct",
    "allo.channel_construct",
}

_STYPE_DIRECTION = {"i": "in", "o": "out", "b": "inout"}

# A generated design is more than its kernels: the backend also emits FIFOs,
# `m_axi` adapters, the control slave and arithmetic cores, and those belong to
# no `@df.kernel`. They are not enumerated by a vendor-specific name list here
# -- which would rot -- but are simply what is left once every kernel has
# claimed its own modules. What IS an error is an unclaimed module that is
# named after one of *this design's* kernels: that means the join missed.


# ---------------------------------------------------------------------------
# Reading the design back out of the IR
# ---------------------------------------------------------------------------


def _find_func(module, name):
    for op in module.body.operations:
        if op.operation.name == "func.func" and op.attributes["sym_name"].value == name:
            return op
    return None


def split_instance(instance, mappings=None):
    """``"pe_0_3"`` -> ``("pe", (0, 3))``.

    A ``@df.kernel(mapping=[...])`` is realized as one function per grid point,
    named ``<kernel>_<i>_<j>...``. When the region's ``mappings`` dict is
    available its keys give the split exactly; a kernel whose own name ends in
    ``_<digits>`` is otherwise indistinguishable from one grid axis, so without
    it we strip every trailing numeric group and say so by construction.
    """
    if mappings:
        for kernel in sorted(mappings, key=len, reverse=True):
            if instance == kernel:
                return kernel, (0,)
            if instance.startswith(kernel + "_"):
                tail = instance[len(kernel) + 1 :]
                if re.fullmatch(r"\d+(_\d+)*", tail):
                    return kernel, tuple(int(p) for p in tail.split("_"))
    base, pid = instance, []
    while True:
        match = re.fullmatch(r"(.+?)_(\d+)", base)
        if match is None:
            break
        base = match.group(1)
        pid.insert(0, int(match.group(2)))
    return base, tuple(pid) if pid else (0,)


def semantic_id(top, kernel, pid):
    """``top/kernel/pid=i,j`` -- the flow's primary key for a PE.

    ``plan_physical_intent`` splits this on ``/`` (first two segments are the
    kernel) and regex-matches ``pid=([0-9,-]+)`` for the grid coordinate, so
    the shape is not ours to vary.
    """
    return f"{top}/{kernel}/pid={','.join(str(p) for p in pid)}"


def _stream_ports(func):
    """The kernel's stream interface: ``(arg_index, ordinal, direction, type)``.

    ``stypes`` is stamped per argument by ``move_stream_to_interface``: ``i``
    read, ``o`` written, ``_`` not a link. The ordinal counts only link
    arguments, because that is the interface the flow zips against the RTL's
    handshake bundles -- an ``m_axi`` memory argument is not one of them.
    """
    stypes = func.attributes["stypes"].value if "stypes" in func.attributes else ""
    ports, ordinal = [], 0
    for index, arg_type in enumerate(func.type.inputs):
        char = stypes[index] if index < len(stypes) else "_"
        if char not in _STYPE_DIRECTION:
            continue
        ports.append((index, ordinal, _STYPE_DIRECTION[char], str(arg_type)))
        ordinal += 1
    return ports


def _memref_args(func):
    """Argument indices of the kernel's non-link (memory) arguments."""
    stypes = func.attributes["stypes"].value if "stypes" in func.attributes else ""
    return [
        index
        for index in range(len(func.type.inputs))
        if (stypes[index] if index < len(stypes) else "_") not in _STYPE_DIRECTION
    ]


def _memref_shape_and_bits(text):
    """``memref<4096xi8>`` -> ``([4096], 8)``; ``memref<56xi64>`` -> ``([56], 64)``."""
    match = re.fullmatch(r"memref<([0-9x]*)([a-z]+)(\d+)>", text)
    if match is None:
        return None, None
    dims = [int(d) for d in match.group(1).split("x") if d]
    return dims, int(match.group(3))


# ---------------------------------------------------------------------------
# The pre-HLS manifest
# ---------------------------------------------------------------------------


def collect(
    module,
    top_func_name,
    top_arguments=None,
    mappings=None,
    backend="vitis",
    clock_period_ns=None,
):
    """Build the ``pre_hls`` manifest from a realized dataflow module.

    ``module`` is the MLIR module after ``allo.dataflow.customize``: one
    ``func.func`` per kernel instance, a top function whose body constructs
    every stream and then calls each instance with the streams it is wired to.
    That body *is* the netlist, so the manifest is a transcription of it.
    """
    top_func = _find_func(module, top_func_name)
    if top_func is None:
        raise RuntimeError(f"top function {top_func_name!r} is not in the module")

    # The links, in declaration order, keyed by the SSA value each one defines.
    link_names, link_types, link_values = [], {}, []
    for op in top_func.entry_block.operations:
        if op.operation.name in _LINK_OPS:
            name = op.attributes["name"].value
            link_names.append(name)
            link_types[name] = str(op.result.type)
            link_values.append((op.result, name))

    def link_of(value):
        for other, name in link_values:
            if other == value:
                return name
        return None

    block_args = list(top_func.arguments)

    def top_arg_of(value):
        for index, arg in enumerate(block_args):
            if arg == value:
                return index
        return None

    pe_instances = []
    channels = {name: [] for name in link_names}
    memory_owner = {}
    for op in top_func.entry_block.operations:
        if op.operation.name != "func.call":
            continue
        instance = str(op.attributes["callee"]).lstrip("@")
        func = _find_func(module, instance)
        if func is None or "df.kernel" not in func.attributes:
            continue
        kernel, pid = split_instance(instance, mappings)
        pe_id = semantic_id(top_func_name, kernel, pid)
        ports = []
        for arg_index, ordinal, direction, type_text in _stream_ports(func):
            stream = link_of(op.operands[arg_index])
            ports.append(
                {
                    "ordinal": ordinal,
                    "channel_id": stream,
                    "stream": stream,
                    "direction": direction,
                    "type": type_text,
                    "argument_index": arg_index,
                }
            )
            if stream is not None:
                channels[stream].append(
                    {
                        "pe": pe_id,
                        "direction": direction,
                        "role": "producer" if direction == "out" else "consumer",
                        "accesses": [{"port_ordinal": ordinal}],
                    }
                )
        for arg_index in _memref_args(func):
            ordinal = top_arg_of(op.operands[arg_index])
            if ordinal is not None:
                memory_owner[ordinal] = (pe_id, instance)
        pe_instances.append(
            {
                "semantic_id": pe_id,
                "kernel": kernel,
                "pid": list(pid),
                "rtl_instance": instance,
                "ports": ports,
            }
        )

    channel_records = [
        {
            "channel_id": name,
            "stream": name,
            "type": link_types[name],
            "endpoints": channels[name],
        }
        for name in link_names
    ]

    arguments = list(top_arguments or [])
    memories = []
    for ordinal, argument in enumerate(arguments):
        owner, instance = memory_owner.get(ordinal, (None, None))
        shape, element_bits = None, None
        if ordinal < len(block_args):
            shape, element_bits = _memref_shape_and_bits(str(block_args[ordinal].type))
        # `postprocess_hls_code` pragmas argument i onto `bundle=gmem{i}`, and
        # Vitis names the adapter it generates for that bundle after it.
        memories.append(
            {
                "ordinal": ordinal,
                "name": argument.get("name"),
                "direction": argument.get("direction"),
                "shape": shape if shape is not None else argument.get("shape"),
                "element_bits": element_bits,
                "bundle": f"gmem{ordinal}" if backend == "vitis" else None,
                "rtl_adapter_module": (
                    f"{top_func_name}_gmem{ordinal}_m_axi"
                    if backend == "vitis"
                    else None
                ),
                "pe": owner,
                "rtl_instance": instance,
            }
        )

    kernels = sorted({pe["kernel"] for pe in pe_instances})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "pre_hls",
        "backend": backend,
        "top": top_func_name,
        "producer": "allo.backend.asic_manifest",
        "top_arguments": arguments,
        "top_interface": top_interface(backend, clock_period_ns),
        "pe_instances": pe_instances,
        "channels": channel_records,
        "memories": memories,
        "kernel_grids": {kernel: _grid(pe_instances, kernel) for kernel in kernels},
        "summary": {
            "unmatched_or_ambiguous": 0,
            "unjoined_post_hls_records": 0,
            "kernels": len(kernels),
            "pe_instances": len(pe_instances),
            "channels": len(channel_records),
            # A stream ARRAY declares an element per grid point, and a chain
            # leaves its terminal elements unwired -- TinyTPU's last column
            # forwards nothing east. Those are real and are kept, because the
            # count moving is how a wiring change shows up.
            "unconnected_channels": sum(
                1 for channel in channel_records if not channel["endpoints"]
            ),
            "memories": len(memories),
        },
    }
    _check(manifest)
    return manifest


def _grid(pe_instances, kernel):
    """The extent of one kernel's ``mapping=``, recovered from its instances."""
    pids = [pe["pid"] for pe in pe_instances if pe["kernel"] == kernel]
    rank = max(len(pid) for pid in pids)
    return [max(pid[axis] for pid in pids) + 1 for axis in range(rank)]


def top_interface(backend, clock_period_ns=None):
    """The control contract the backend gives the top module."""
    if backend == "vitis":
        interface = {
            "protocol": "vitis_ap_ctrl",
            "clock": {"name": "ap_clk", "edge": "rising"},
            "reset": {
                "name": "ap_rst_n",
                "polarity": "active_low",
                "default_asserted_cycles": 2,
            },
            "completion": {"kind": "ap_done", "port": "ap_done", "active_level": 1},
        }
    elif backend == "systemc":
        interface = {
            "protocol": "systemc_connections",
            "clock": {"name": "clk", "edge": "rising"},
            "reset": {
                "name": "rst",
                "polarity": "active_low",
                "default_asserted_cycles": 2,
            },
            "completion": {"kind": "top_done", "port": "done", "active_level": 1},
        }
    else:
        interface = {
            "protocol": "catapult_direct_array",
            "clock": {"name": "clk", "edge": "rising"},
            "reset": {
                "name": "rst",
                "polarity": "active_high",
                "default_asserted_cycles": 2,
            },
            "completion": {"kind": "per_argument_triosy", "active_level": 1},
        }
    if clock_period_ns:
        interface["clock"]["period_ns"] = float(clock_period_ns)
    return interface


def _check(manifest):
    """The invariants the flow's planners assume but do not all state.

    Every one of these is a `KeyError`, a `StopIteration` or a silent
    mis-placement several nodes downstream if it is violated, so it is checked
    where the manifest is written rather than where it is read.
    """
    ids = [pe["semantic_id"] for pe in manifest["pe_instances"]]
    duplicates = {name for name in ids if ids.count(name) > 1}
    if duplicates:
        raise RuntimeError(f"duplicate PE semantic_id: {sorted(duplicates)}")
    known = set(ids)
    for pe in manifest["pe_instances"]:
        ordinals = [port["ordinal"] for port in pe["ports"]]
        if ordinals != list(range(len(ordinals))):
            raise RuntimeError(
                f"{pe['semantic_id']}: stream port ordinals must be dense and "
                f"start at 0, got {ordinals}"
            )
    for channel in manifest["channels"]:
        roles = [endpoint["role"] for endpoint in channel["endpoints"]]
        # Allo allows one writer and one reader per stream, and the flow's
        # placer takes the first non-self endpoint as *the* peer -- a third
        # endpoint would be silently dropped rather than rejected.
        if roles.count("producer") > 1 or roles.count("consumer") > 1:
            raise RuntimeError(
                f"channel {channel['channel_id']} has endpoints {roles}; a "
                "stream has at most one writer and one reader"
            )
        if roles and sorted(roles) != ["consumer", "producer"]:
            raise RuntimeError(
                f"channel {channel['channel_id']} is half-wired: {roles}"
            )
        for endpoint in channel["endpoints"]:
            if endpoint["pe"] not in known:
                raise RuntimeError(
                    f"channel {channel['channel_id']} names unknown PE "
                    f"{endpoint['pe']}"
                )
            for access in endpoint["accesses"]:
                pe = next(
                    item
                    for item in manifest["pe_instances"]
                    if item["semantic_id"] == endpoint["pe"]
                )
                if not any(
                    port["ordinal"] == access["port_ordinal"] for port in pe["ports"]
                ):
                    raise RuntimeError(
                        f"channel {channel['channel_id']} names ordinal "
                        f"{access['port_ordinal']} that {endpoint['pe']} "
                        f"does not have"
                    )


# ---------------------------------------------------------------------------
# Joining the manifest to the RTL the backend produced
# ---------------------------------------------------------------------------

_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", re.M)


def read_rtl_modules(rtl_dir):
    """``{module name: source text}`` for every module in a backend RTL dir."""
    modules = {}
    for entry in sorted(os.listdir(rtl_dir)):
        if not entry.endswith((".v", ".sv")) or entry.endswith("_stub.v"):
            continue
        path = os.path.join(rtl_dir, entry)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        for name in _MODULE_RE.findall(text):
            modules.setdefault(name, text)
    return modules


def _equivalence_hash(text, tokens):
    """Hash a module's text with its own identity spelled out of it.

    Two generated modules are the same hardware if they differ only in the
    names Vitis derived from the instance. Replacing those tokens before
    hashing is what makes "these two PEs are the same macro" a claim about the
    RTL rather than about the names.
    """
    normalized = text
    for token in sorted(tokens, key=len, reverse=True):
        normalized = normalized.replace(token, "\x00INSTANCE\x00")
    normalized = re.sub(r"//.*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def enrich(pre, rtl_dir, published_path=None):
    """Join the pre-HLS records to the emitted RTL -> ``post_hls_enriched``.

    Vitis emits one module per kernel instance, named ``<top>_<instance>``
    (sometimes with a trailing disambiguator), plus ``<top>_<instance>_...``
    children for each pipelined loop it split out, plus shared support modules
    that belong to no kernel. This walks that naming and records, per PE, the
    root module it became, the children it owns, and a hash of the root that is
    stable under Vitis's instance-derived renaming -- which is what turns a set
    of PEs into a set of macro classes.
    """
    top = pre["top"]
    modules = read_rtl_modules(rtl_dir)

    claimed = set()
    roots = {}
    ambiguous = []
    for pe in pre["pe_instances"]:
        instance = pe.get("rtl_instance") or pe["semantic_id"].rsplit("/", 1)[0]
        exact = f"{top}_{instance}"
        if exact in modules:
            roots[pe["semantic_id"]] = exact
            continue
        # Vitis appends a numeric disambiguator to a function it had to clone.
        pattern = re.compile(r"^" + re.escape(exact) + r"_\d+$")
        candidates = sorted(name for name in modules if pattern.fullmatch(name))
        candidates = [
            name
            for name in candidates
            if not any(
                name == f"{top}_{other.get('rtl_instance')}"
                for other in pre["pe_instances"]
            )
        ]
        if len(candidates) == 1:
            roots[pe["semantic_id"]] = candidates[0]
        else:
            ambiguous.append((pe["semantic_id"], candidates))

    pe_instances = []
    for pe in pre["pe_instances"]:
        record = dict(pe)
        root = roots.get(pe["semantic_id"])
        if root is None:
            record["post_hls_records"] = []
            pe_instances.append(record)
            continue
        claimed.add(root)
        children = sorted(
            name for name in modules if name.startswith(root + "_") and name != root
        )
        claimed.update(children)
        tokens = {root, pe.get("rtl_instance") or ""} - {""}
        record["post_hls_records"] = [
            {
                "rtl_root_module": root,
                "rtl_modules": [{"name": name} for name in [root] + children],
                "rtl_equivalence_hash": _equivalence_hash(modules[root], tokens),
            }
        ]
        pe_instances.append(record)

    # Macro classes: PEs of one kernel whose root RTL hashes to the same thing.
    classes = {}
    for pe in pe_instances:
        records = pe["post_hls_records"]
        if not records:
            continue
        key = (pe["kernel"], records[0]["rtl_equivalence_hash"])
        classes.setdefault(key, []).append(pe)
    macro_groups = []
    for (kernel, digest), members in sorted(
        classes.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        macro_groups.append(
            {
                "macro_class_id": f"{kernel}_{digest[:10]}",
                "representative": members[0]["semantic_id"],
                "member_count": len(members),
                "members": [
                    {
                        "semantic_id": member["semantic_id"],
                        "rtl_module": member["post_hls_records"][0]["rtl_root_module"],
                        "orientation": "unassigned",
                    }
                    for member in members
                ],
                "proof": {
                    "status": "proven",
                    "method": "normalized_rtl_module_text",
                    "implementation_contract_hash": digest,
                },
                "rtl_audit": {
                    "authority": True,
                    "status": "agree",
                    "distinct_hashes": [digest],
                },
            }
        )

    # Anything a kernel did not claim is either backend plumbing or a module
    # the join should have found. Only the second is an error, and the design's
    # own kernel names are what tell them apart.
    kernel_prefixes = tuple(
        f"{top}_{kernel}" for kernel in {pe["kernel"] for pe in pre["pe_instances"]}
    )
    unattributed = sorted(
        name for name in modules if name not in claimed and name != top
    )
    unclaimed = [name for name in unattributed if name.startswith(kernel_prefixes)]
    final = dict(pre)
    final.update(
        {
            "stage": "post_hls_enriched",
            "pe_instances": pe_instances,
            "macro_groups": macro_groups,
            "rtl_artifact": {
                "published_path": published_path
                or (
                    "backend-rtl"
                    if pre.get("backend") == "vitis"
                    else "backend-rtl/concat_rtl.v"
                ),
                "module_count": len(modules),
            },
            "support_modules": [name for name in unattributed if name not in unclaimed],
        }
    )
    summary = dict(pre.get("summary", {}))
    summary.update(
        {
            "unmatched_or_ambiguous": len(ambiguous),
            "unjoined_post_hls_records": len(unclaimed),
            "macro_classes": len(macro_groups),
            "rtl_modules": len(modules),
        }
    )
    if ambiguous:
        summary["ambiguous_pe_instances"] = [name for name, _ in ambiguous]
    if unclaimed:
        summary["unclaimed_rtl_modules"] = unclaimed
    final["summary"] = summary
    return final


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _tcl_key(text):
    return re.sub(r"[^A-Za-z0-9_.:/=,-]+", "_", str(text))


def to_tcl(manifest):
    """The same records as Tcl arrays, for the flow's Tcl-side nodes."""
    lines = [
        "# Generated by allo.backend.asic_manifest -- do not edit.",
        f"set allo_asic(schema_version) {manifest['schema_version']}",
        f"set allo_asic(stage) {manifest['stage']}",
        f"set allo_asic(backend) {manifest.get('backend', 'vitis')}",
        f"set allo_asic(top) {manifest['top']}",
        f"set allo_asic(pe_count) {len(manifest.get('pe_instances', []))}",
        f"set allo_asic(channel_count) {len(manifest.get('channels', []))}",
    ]
    for pe in manifest.get("pe_instances", []):
        key = _tcl_key(pe["semantic_id"])
        lines.append(f"set allo_asic_pe_kernel({key}) {pe['kernel']}")
        lines.append(
            f"set allo_asic_pe_pid({key}) {{{' '.join(str(p) for p in pe['pid'])}}}"
        )
        records = pe.get("post_hls_records") or []
        if records:
            lines.append(f"set allo_asic_pe_rtl({key}) {records[0]['rtl_root_module']}")
    for group in manifest.get("macro_groups", []):
        key = _tcl_key(group["macro_class_id"])
        lines.append(f"set allo_asic_macro_members({key}) {group['member_count']}")
        lines.append(
            f"set allo_asic_macro_modules({key}) "
            f"{{{' '.join(m['rtl_module'] for m in group['members'])}}}"
        )
    return "\n".join(lines) + "\n"


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write(manifest, project, name, debug_dir=None):
    """Write ``<name>.json`` and ``<name>.tcl`` under ``project``."""
    os.makedirs(project, exist_ok=True)
    stem = name[:-5] if name.endswith(".json") else name
    _write(
        os.path.join(project, f"{stem}.json"),
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
    )
    _write(os.path.join(project, f"{stem}.tcl"), to_tcl(manifest))
    if debug_dir:
        write_debug(manifest, os.path.join(project, debug_dir))
    return os.path.join(project, f"{stem}.json")


def write_debug(manifest, directory):
    """Flat, greppable views of the same records, for reading by eye."""
    os.makedirs(directory, exist_ok=True)
    rows = ["semantic_id\tkernel\tpid\trtl_instance\tports"]
    for pe in manifest.get("pe_instances", []):
        rows.append(
            "\t".join(
                [
                    pe["semantic_id"],
                    pe["kernel"],
                    ",".join(str(p) for p in pe["pid"]),
                    str(pe.get("rtl_instance")),
                    str(len(pe.get("ports", []))),
                ]
            )
        )
    _write(os.path.join(directory, "pe-instances.tsv"), "\n".join(rows) + "\n")
    rows = ["channel_id\ttype\tproducer\tconsumer"]
    for channel in manifest.get("channels", []):
        ends = channel.get("endpoints", [])
        producer = next((e["pe"] for e in ends if e.get("role") == "producer"), "")
        consumer = next((e["pe"] for e in ends if e.get("role") == "consumer"), "")
        rows.append(
            "\t".join(
                [channel["channel_id"], channel.get("type", ""), producer, consumer]
            )
        )
    _write(os.path.join(directory, "channels.tsv"), "\n".join(rows) + "\n")
    _write(
        os.path.join(directory, "summary.json"),
        json.dumps(manifest.get("summary", {}), indent=2) + "\n",
    )


# ---------------------------------------------------------------------------
# The `configs["asic_manifest"]` contract
# ---------------------------------------------------------------------------


def options(configs):
    """``configs["asic_manifest"]`` if it asks for a manifest, else ``None``."""
    requested = (configs or {}).get("asic_manifest")
    if not isinstance(requested, dict) or not requested.get("enabled", False):
        return None
    return {
        "path": requested.get("path", "asic-manifest.json"),
        "debug_artifacts": requested.get("debug_artifacts", False),
        "debug_dir": requested.get("debug_dir", "asic-debug"),
    }


def final_name(path):
    """``asic-manifest.json`` -> ``asic-manifest-final`` (the node's names)."""
    stem = path[:-5] if path.endswith(".json") else path
    return f"{stem}-final"


def rtl_directory(backend, project):
    """Where the backend left the RTL it just synthesized.

    The same layouts ``nodes/allo-asic-compilation/backend.py`` publishes from,
    so the manifest is enriched against exactly the files the flow will read.
    """
    import glob  # noqa: PLC0415

    if backend == "vitis":
        candidates = sorted(
            path
            for path in glob.glob(os.path.join(project, "*.prj", "*", "syn", "verilog"))
            if os.path.isdir(path)
        )
    else:
        candidates = sorted(
            {
                os.path.dirname(path)
                for name in ("concat_rtl.v", "rtl.v")
                for path in glob.glob(os.path.join(project, "**", name), recursive=True)
                if os.path.isfile(path)
            }
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one {backend} RTL directory under {project}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def emit_final(project, configs, backend):
    """Enrich the pre-HLS manifest with the RTL that has just been produced.

    Called by the backend once synthesis has succeeded: only then do the
    module names exist to join the architecture to. No-op unless ``configs``
    asked for a manifest and the pre-HLS one is where it was written.
    """
    requested = options(configs)
    if requested is None:
        return None
    pre_path = os.path.join(project, requested["path"])
    if not os.path.isfile(pre_path):
        raise RuntimeError(
            f"asic_manifest was requested but {pre_path} is missing; the "
            "pre-HLS manifest is written at build() time"
        )
    with open(pre_path, "r", encoding="utf-8") as handle:
        pre = json.load(handle)
    final = enrich(pre, rtl_directory(backend, project))
    write(
        final,
        project,
        final_name(requested["path"]),
        debug_dir=requested["debug_dir"] if requested["debug_artifacts"] else None,
    )
    return final
