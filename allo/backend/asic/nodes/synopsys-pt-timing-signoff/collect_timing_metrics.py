#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

def slack(path):
    text = path.read_text(errors='replace') if path.is_file() else ''
    values = [float(v) for v in re.findall(r'slack\s+\((?:MET|VIOLATED)\)\s+([-+0-9.eE]+)', text, re.I)]
    return min(values) if values else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wall-seconds', type=float, required=True)
    p.add_argument('--setup-target', type=float, required=True)
    p.add_argument('--hold-target', type=float, required=True)
    p.add_argument('--policy', choices=('error', 'report'), required=True)
    a = p.parse_args()
    setup_files = list(Path('reports').glob('*.timing.setup.rpt'))
    hold_files = list(Path('reports').glob('*.timing.hold.rpt'))
    setup = slack(setup_files[0]) if setup_files else None
    hold = slack(hold_files[0]) if hold_files else None
    if setup is None or hold is None: raise SystemExit('missing parseable setup/hold timing reports')
    met = setup >= a.setup_target and hold >= a.hold_target
    result = {'schema_version': 1, 'node': 'synopsys-pt-timing-signoff',
              'status': 'passed' if met else 'reported' if a.policy == 'report' else 'failed',
              'wall_seconds': a.wall_seconds, 'policy': a.policy,
              'setup': {'wns_ns': setup, 'target_slack_ns': a.setup_target, 'target_met': setup >= a.setup_target},
              'hold': {'wns_ns': hold, 'target_slack_ns': a.hold_target, 'target_met': hold >= a.hold_target}}
    Path('outputs/timing-metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    if not met and a.policy == 'error': raise SystemExit('timing targets were not met')

if __name__ == '__main__': main()
