import importlib.util, json
from pathlib import Path

PATH = Path(__file__).with_name('summarize_flow.py')
SPEC = importlib.util.spec_from_file_location('flat_summary', PATH)
SUMMARY = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(SUMMARY)

def test_summary_accepts_optional_stages(tmp_path, monkeypatch):
    inputs, outputs = tmp_path/'inputs', tmp_path/'outputs'; inputs.mkdir()
    (inputs/'synthesis-metrics.json').write_text(json.dumps({'status': 'passed'}))
    (inputs/'sram-contract.json').write_text(json.dumps({'num_srams': 0, 'srams': []}))
    monkeypatch.setattr(SUMMARY, 'INPUTS', inputs); monkeypatch.setattr(SUMMARY, 'OUTPUTS', outputs)
    monkeypatch.setenv('design_name', 'gcd')
    SUMMARY.main()
    result = json.loads((outputs/'flow-summary.json').read_text())
    assert result['run']['implementation_style'] == 'flat'
    assert result['stages']['synthesis']['status'] == 'passed'
    assert result['stages']['lvs']['status'] == 'unavailable'
    assert result['sram_contract']['num_srams'] == 0
    assert 'SRAMs: 0' in (outputs/'flow-summary.txt').read_text()
    assert 'set asic_flow_num_srams {0}' in (outputs/'flow-summary.tcl').read_text()
