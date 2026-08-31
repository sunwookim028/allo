#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors='replace')
antenna_policy, drc_policy = sys.argv[3:5]
if antenna_policy not in {'error', 'report', 'off'}: raise SystemExit('invalid antenna policy')
if drc_policy not in {'error', 'report'}: raise SystemExit('invalid DRC policy')
total_match = re.search(r'TOTAL DRC Results Generated:\s*(\d+)', text, re.I)
if not total_match: raise SystemExit('Calibre summary has no total result count')
global_part = re.split(r'RULECHECK RESULTS STATISTICS \(BY CELL\)', text, maxsplit=1, flags=re.I)[0]
counts = [(n, int(c)) for n, c in re.findall(
    r'RULECHECK\s+(\S+)\s+.*?TOTAL Result Count\s*=\s*(\d+)', global_part, re.I)]
if not counts: raise SystemExit('Calibre summary has no per-rule counts')
antenna = sum(c for n, c in counts if n.lower().startswith('antenna.'))
other = sum(c for n, c in counts if not n.lower().startswith('antenna.'))
failed = (drc_policy == 'error' and other) or (antenna_policy == 'error' and antenna)
result = {'schema_version': 1, 'total_results': int(total_match.group(1)),
          'antenna_results': antenna, 'non_antenna_results': other,
          'antenna_check_policy': antenna_policy, 'drc_check_policy': drc_policy,
          'status': 'failed' if failed else 'reported' if antenna or other else 'passed'}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
if failed: raise SystemExit('Calibre DRC results violate the selected policy')
