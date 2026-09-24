#!/usr/bin/env python3
"""Prepare an Innovus timing-constraint export for PrimeTime.

Innovus's signoff interpretation keeps generated zero-delay primary inputs at
the chip boundary when clock source-latency compensation is present. PrimeTime
otherwise applies that source latency to the input-data launch reference. Mark
each exported input delay as already containing source latency so both tools
analyze the same external-input relationship.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SET_INPUT_DELAY = re.compile(r"^(?P<indent>\s*)set_input_delay(?P<rest>\s+.*)$")


def prepare_sdc(text: str) -> str:
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        match = SET_INPUT_DELAY.match(body)
        if match and "-source_latency_included" not in match.group("rest"):
            body = (
                f"{match.group('indent')}set_input_delay "
                f"-source_latency_included{match.group('rest')}"
            )
        output.append(body + ending)
    return "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_sdc", type=Path)
    parser.add_argument("output_sdc", type=Path)
    args = parser.parse_args()

    source = args.input_sdc.read_text()
    prepared = prepare_sdc(source)
    args.output_sdc.write_text(prepared)

    input_count = sum(
        1 for line in source.splitlines() if SET_INPUT_DELAY.match(line)
    )
    prepared_count = prepared.count("-source_latency_included")
    if input_count == 0:
        raise RuntimeError("input SDC contains no set_input_delay commands")
    if prepared_count < input_count:
        raise RuntimeError(
            "not every set_input_delay command has source-latency semantics"
        )
    print(
        f"Prepared {args.output_sdc}: marked {input_count} input-delay commands "
        "as source-latency included"
    )


if __name__ == "__main__":
    main()
