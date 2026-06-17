#!/usr/bin/env python

import argparse
from pathlib import Path
import yaml


def bool_text(value):
    return str(value).lower() in ("1", "true", "yes", "on")

def render(template, values):
    text = template
    for key, value in values.items():
        text = text.replace("{{ " + key + " }}", str(value))
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tech-name", required=True)
    parser.add_argument("--process-corner", required=True)
    parser.add_argument("--supply-voltage", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--check-lvsdrc", required=True)
    parser.add_argument("--route-supplies", required=True)
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    template = Path(args.template).read_text()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sram in manifest.get("srams", []):
        values = {
            "name": sram["name"],
            "word_size": sram["word_size"],
            "num_words": sram["num_words"],
            "num_banks": sram.get("num_banks", 1),
            "words_per_row": sram.get("words_per_row", 4),
            "write_size": sram.get("write_size", sram["word_size"]),
            "num_rw_ports": sram.get("num_rw_ports", 1),
            "num_r_ports": sram.get("num_r_ports", 0),
            "num_w_ports": sram.get("num_w_ports", 0),
            "tech_name": args.tech_name,
            "process_corner": args.process_corner,
            "supply_voltage": args.supply_voltage,
            "temperature": args.temperature,
            "check_lvsdrc": bool_text(args.check_lvsdrc),
            "route_supplies": bool_text(args.route_supplies),
        }
        cfg = render(template, values)
        (out_dir / f"{sram['name']}_cfg.py").write_text(cfg)

if __name__ == "__main__":
    main()