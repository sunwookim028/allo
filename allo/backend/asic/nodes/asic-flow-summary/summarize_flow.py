#!/usr/bin/env python3
"""Create a local, dependency-free summary of a flat RTL-to-GDS run."""
import json, os, re
from datetime import datetime, timezone
from pathlib import Path

INPUTS, OUTPUTS = Path('inputs'), Path('outputs')
STAGES = {
    'synthesis': 'synthesis-metrics.json', 'pnr': 'pnr-metrics.json',
    'timing': 'timing-metrics.json', 'gdsmerge': 'gdsmerge-metrics.json',
    'drc': 'drc-metrics.json', 'lvs': 'lvs-metrics.json',
    'rtl_simulation': 'rtl-simulation-report.json',
    'ffgl_simulation': 'ffgl-simulation-report.json',
    'bagl_simulation': 'bagl-simulation-report.json',
}

def optional_json(name):
    path = INPUTS / name
    if not path.is_file(): return {'status': 'unavailable', 'source': name}
    try: value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {'status': 'invalid', 'source': name, 'error': str(error)}
    value['source'] = name
    return value

def power_total(text):
    patterns = [r'Total\s+Dynamic\s+Power\s*=\s*([-+0-9.eE]+)',
                r'Total\s+Power\s*=\s*([-+0-9.eE]+)']
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match: return float(match.group(1))
    return None

def main():
    OUTPUTS.mkdir(exist_ok=True)
    stages = {key: optional_json(name) for key, name in STAGES.items()}
    power_path = INPUTS / 'power.rpt'
    power = {'status': 'available', 'total_reported_power': power_total(power_path.read_text(errors='replace'))} if power_path.is_file() else {'status': 'unavailable'}
    result = {
        'schema_version': 1,
        'run': {'design_name': os.environ.get('design_name', 'undefined'),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'implementation_style': 'flat'},
        'stages': stages, 'power': power,
        'activity_source': optional_json('activity-source.json'),
        'drc_policy': optional_json('drc-policy.json'),
    }
    OUTPUTS.joinpath('flow-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [f"ASIC flow summary: {result['run']['design_name']}", 'Implementation: flat']
    lines += [f"{name}: {data.get('status', 'unknown')}" for name, data in stages.items()]
    lines.append(f"Power: {power.get('total_reported_power', 'unavailable')}")
    OUTPUTS.joinpath('flow-summary.txt').write_text('\n'.join(lines) + '\n')
    tcl = [f"set asic_flow_design_name {{{result['run']['design_name']}}}", 'set asic_flow_implementation_style flat']
    tcl += [f"set asic_flow_{name}_status {{{data.get('status', 'unknown')}}}" for name, data in stages.items()]
    OUTPUTS.joinpath('flow-summary.tcl').write_text('\n'.join(tcl) + '\n')

if __name__ == '__main__': main()
