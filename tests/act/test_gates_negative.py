# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure paths of the two push-button gates.

A check that has only ever said ok proves nothing. Every test here constructs
the failure -- a mismatched ``QD``, a model that should not map, a missing
reference, an export re-emitted after its area was measured -- and asserts both
that the gate refuses AND that it says why. The message is part of the
contract: a gate that fails without naming the cause is one nobody can act on.

Each test carries a positive control in the same shape, because a negative test
that would also pass against a gate which refuses everything is not evidence.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    """Upward search for a marker, never a count of levels."""
    d = HERE
    while True:
        if (os.path.isdir(os.path.join(d, "examples", "tinytpu"))
                and os.path.isdir(os.path.join(d, "allo", "backend", "asic"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("not inside an allo checkout")
        d = parent


REPO = _repo_root()
WORKLOAD_GATE = os.path.join(REPO, "examples", "tinytpu", "workloads", "gate.py")
CLAIMS = os.path.join(REPO, "examples", "tinytpu", "workloads", "claims.json")
SPECS = os.path.join(REPO, "examples", "tinytpu", "workloads", "specs")
PAIRING_GATE = os.path.join(REPO, "allo", "backend", "asic", "tools",
                            "check_pairing.py")
REPORTS = os.path.join(REPO, "examples", "tinytpu", "asic_synthesis", "reports")
EXPORTS = os.path.join(REPO, "dev", "records", "tinytpu", "rtl_handoff")
PAIRINGS = os.path.join(REPO, "examples", "tinytpu", "asic_synthesis",
                        "pairings.json")

pytest.importorskip("torch", reason="the suite's front end is torch.fx")

#: Two models is enough to exercise every branch and keeps each run ~2 s.
SOME = ["mlp_tiny", "mlp_bias"]


def run(argv, build=None):
    env = dict(os.environ, PYTHONPATH=REPO)
    for key in [k for k in env if k.startswith("TPU_")]:
        del env[key]
    env.update(build or {})
    out = subprocess.run([sys.executable] + argv, capture_output=True,
                         text=True, env=env, cwd=REPO, timeout=1800)
    return out.returncode, out.stdout + out.stderr


def workload_gate(claims, models=SOME, docs=REPO, specs=SPECS, build=None):
    argv = [WORKLOAD_GATE, "--claims", claims, "--docs", docs,
            "--specs", specs, "--models"] + list(models)
    return run(argv, build or {})


def pairing_gate(pairings, exports=EXPORTS, reports=REPORTS):  # noqa: D401
    return run([PAIRING_GATE, "--reports", reports, "--exports", exports,
                "--pairings", pairings])


def mutate(tmp_path, name, edit):
    """A copy of a declaration file with one thing changed."""
    source = CLAIMS if name == "claims.json" else PAIRINGS
    with open(source) as fh:
        data = json.load(fh)
    edit(data)
    path = str(tmp_path / name)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return path


# ---------------------------------------------------------------- gate 1


def test_the_workload_gate_passes_unmodified():
    """The positive control. Without it the refusals below prove nothing."""
    code, text = workload_gate(CLAIMS)
    assert code == 0, text
    assert "WORKLOAD GATE OK" in text
    assert "T=4 MAXDIM=64 QD=16 DMA_WORDS=1" in text, (
        "the gate must print the configuration it ran at, or 'OK' names no "
        "machine")


def test_a_measurement_from_a_different_QD_cannot_be_paired(tmp_path):
    """The headline refusal: a number measured on another build."""
    def edit(data):
        for run_ in data["models"]["mlp_tiny"]["measured"]:
            run_["config"]["QD"] = 8
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit))
    assert code == 1, text
    assert "CONFIRMED" in text and "QD=8 not 16" in text, text


def test_a_measurement_with_no_QD_at_all_is_cannot_pair_not_matches(tmp_path):
    """An older row carries no QD field, and the absence dates it.

    This is the case the gate must not get wrong: a missing key reads as
    agreement to anything comparing dictionaries loosely.
    """
    def edit(data):
        for run_ in data["models"]["mlp_tiny"]["measured"]:
            run_["config"].pop("QD")
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit))
    assert code == 1, text
    assert "omits QD" in text, text
    assert "cannot pair" in text.lower(), text


def test_a_model_that_should_not_map_but_does_is_a_failure(tmp_path):
    """mlp_bias is in the suite deliberately to NOT map.

    Declaring it clean is the shape of the mistake: a gate that made every
    model pass would be the wrong gate.
    """
    def edit(data):
        data["models"]["mlp_bias"]["refusals"] = []
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit))
    assert code == 1, text
    assert "refusal" in text.lower(), text


def test_a_refusal_that_changes_its_reason_is_a_failure(tmp_path):
    def edit(data):
        data["models"]["mlp_bias"]["refusals"][0]["why_contains"] = \
            "the ISA has a bias epilogue"
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit))
    assert code == 1, text
    assert "no longer says" in text, text


def test_a_model_in_the_suite_with_no_claim_is_a_failure(tmp_path):
    def edit(data):
        del data["models"]["mlp_wide"]
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit))
    assert code == 1, text
    assert "mlp_wide is in the suite and declares no claim" in text, text


def test_a_missing_claims_file_fails_closed(tmp_path):
    code, text = workload_gate(str(tmp_path / "there-is-no-such-file.json"))
    assert code == 1, text
    assert "unreadable" in text, text
    assert "must fail, not pass" in text, text


def test_correct_but_not_confirmed_may_not_carry_a_measurement(tmp_path):
    """mlp_wide is correct and NOT confirmed. The ladder is not decoration."""
    def edit(data):
        data["models"]["mlp_wide"]["measured"] = [{
            "config": {"T": 4, "MAXDIM": 64, "QD": 16, "DMA_WORDS": 1},
            "per_layer": {"mlp_wide_l0": 30000, "mlp_wide_l1": 28000},
            "total": 58000,
        }]
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit),
                               models=["mlp_wide"])
    assert code == 1, text
    assert "declared CORRECT" in text, text


def test_not_confirmed_must_say_why(tmp_path):
    def edit(data):
        data["models"]["mlp_wide"]["not_confirmed"] = []
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit),
                               models=["mlp_wide"])
    assert code == 1, text
    assert "does not say why" in text, text


def test_per_layer_cycles_must_sum_to_the_model_total(tmp_path):
    """A model figure is a sum over layers with no fusion and no residency."""
    def edit(data):
        data["models"]["mlp_tiny"]["measured"][0]["total"] = 1149
    code, text = workload_gate(mutate(tmp_path, "claims.json", edit))
    assert code == 1, text
    assert "sum to 1150" in text, text


def test_a_page_that_drops_the_measurement_is_a_failure(tmp_path):
    """The association is declared, so the page is told what it must carry."""
    docs = tmp_path / "docs-copy"
    page = os.path.join("docs", "source", "designs", "workload_suite.rst")
    os.makedirs(str(docs / os.path.dirname(page)))
    with open(os.path.join(REPO, page)) as fh:
        text = fh.read()
    with open(str(docs / page), "w") as fh:
        fh.write(text.replace("**1 150**", "**1 234**"))
    code, out = workload_gate(CLAIMS, docs=str(docs))
    assert code == 1, out
    assert "no longer contains '**1 150**'" in out, out


def test_a_committed_spec_that_drifts_from_the_extractor_is_a_failure(tmp_path):
    specs = str(tmp_path / "specs")
    shutil.copytree(SPECS, specs)
    path = os.path.join(specs, "mlp_tiny_l0.json")
    with open(path) as fh:
        spec = json.load(fh)
    spec["dims"]["n"] = 8
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    code, text = workload_gate(CLAIMS, specs=specs)
    assert code == 1, text
    assert "is not what the extractor emits now" in text, text


# ---------------------------------------------------------------- gate 2


def test_the_pairing_gate_passes_unmodified():
    """The positive control, and it must admit something.

    A join that admitted nothing would pass this file's other tests while
    proving only that it refuses.
    """
    code, text = pairing_gate(PAIRINGS)
    assert code == 0, text
    assert "PAIRING OK" in text
    assert "ADMITTED" in text and "um2" in text


def test_an_export_re_emitted_after_its_area_run_breaks_the_pair(tmp_path):
    """The T8 case, reproduced deliberately on a variant that currently passes.

    An area figure describes the RTL that was synthesised, not the RTL that
    happens to sit at the same path now.
    """
    exports = str(tmp_path / "rtl_handoff")
    shutil.copytree(EXPORTS, exports)
    manifest = os.path.join(exports, "T4_MAXDIM64_shipped", "sv2v_manifest.f")
    with open(manifest, "a") as fh:
        fh.write("# a line added after the synthesis run\n")
    code, text = pairing_gate(PAIRINGS, exports=exports)
    assert code == 1, text
    assert "re-emitted after the run" in text, text


def test_the_QD_16_published_row_may_not_be_paired_with_the_QD_8_area(tmp_path):
    """The one-line edit nothing else in the tree would catch.

    175 / 265 / 421 / 482 / 674 is a TPU_QD=16 row; the MAXDIM=16 area was
    synthesised from an export that predates the default moving and records no
    QD. Promoting that refusal to a claim must fail.
    """
    def edit(data):
        moved = [p for p in data["refused"]
                 if "current published row" in p["name"]]
        assert len(moved) == 1
        data["refused"].remove(moved[0])
        moved[0].pop("expect")
        data["pairings"].append(moved[0])
    code, text = pairing_gate(mutate(tmp_path, "pairings.json", edit))
    assert code == 1, text
    assert "CLAIMED PAIR REFUSED" in text, text
    assert "records no QD" in text, text


def test_the_workload_cycles_may_not_be_paired_with_any_committed_area(tmp_path):
    """The end-to-end pair the project wants, and cannot have yet."""
    def edit(data):
        moved = [p for p in data["refused"]
                 if "workload suite's model cycles" in p["name"]]
        assert len(moved) == 1
        data["refused"].remove(moved[0])
        moved[0].pop("expect")
        data["pairings"].append(moved[0])
    code, text = pairing_gate(mutate(tmp_path, "pairings.json", edit))
    assert code == 1, text
    assert "records no QD" in text, text


def test_a_refusal_whose_cause_is_repaired_must_be_promoted(tmp_path):
    """The allow-list is re-derived, not trusted.

    Moving an admissible pair into `refused` is the same situation as a
    refusal whose cause was fixed and whose entry was left behind: the gate
    must notice and say so rather than quietly keep refusing.
    """
    def edit(data):
        moved = [p for p in data["pairings"]
                 if "burst-widened" in p["name"]][0]
        data["pairings"].remove(moved)
        moved["expect"] = "missing-key"
        moved["why"] = "a reason that no longer applies"
        data["refused"].append(moved)
    code, text = pairing_gate(mutate(tmp_path, "pairings.json", edit))
    assert code == 1, text
    assert "now ADMISSIBLE" in text, text


def test_a_refusal_with_the_wrong_stated_cause_is_a_failure(tmp_path):
    def edit(data):
        for p in data["refused"]:
            if p["name"].startswith("T8 "):
                p["expect"] = "missing-key"
    code, text = pairing_gate(mutate(tmp_path, "pairings.json", edit))
    assert code == 1, text
    assert "refused for 'export-superseded'" in text, text
    assert "misleading" in text, text


def test_a_refusal_with_no_reason_is_a_failure(tmp_path):
    def edit(data):
        for p in data["refused"]:
            if p["name"].startswith("T8 "):
                p["why"] = ""
    code, text = pairing_gate(mutate(tmp_path, "pairings.json", edit))
    assert code == 1, text
    assert "carries no reason" in text, text


def test_an_area_figure_with_no_declared_status_is_a_failure(tmp_path):
    """Silence is how an unpaired figure comes to be quoted."""
    def edit(data):
        data["refused"] = [p for p in data["refused"]
                           if p["area"]["variant"] != "gemmini_DIM8_full"]
    code, text = pairing_gate(mutate(tmp_path, "pairings.json", edit))
    assert code == 1, text
    assert "COVERAGE" in text and "gemmini_DIM8_full" in text, text


def test_an_unreadable_pairings_file_fails_closed(tmp_path):
    path = str(tmp_path / "broken.json")
    with open(path, "w") as fh:
        fh.write("{ not json")
    code, text = pairing_gate(path)
    assert code == 1, text
    assert "unreadable" in text and "must fail, not pass" in text, text


def test_a_missing_reports_directory_fails_closed(tmp_path):
    code, text = pairing_gate(PAIRINGS, reports=str(tmp_path / "nothing"))
    assert code == 1, text
    assert "no reports directory" in text, text


def test_an_area_run_with_no_settings_snapshot_cannot_be_paired():
    """superseded_export_T4_MAXDIM16 records nothing about the RTL it read."""
    code, text = pairing_gate(PAIRINGS)
    assert code == 0, text
    assert "the superseded MAXDIM=16 export: the area run recorded no " \
           "settings snapshot" in text, text


def test_the_published_MAXDIM_16_build_has_no_declared_workload_measurement():
    """The mistake the project makes most often, refused rather than absorbed.

    TPU_MAXDIM defaults to 64 and the published five-shape row is a MAXDIM=16
    measurement, so a shell that has pinned MAXDIM=16 for reproduce.sh is a
    different machine from the one the workload numbers were taken on. The
    gate must not quote them there.
    """
    code, text = workload_gate(CLAIMS, build={"TPU_MAXDIM": "16"})
    assert code == 1, text
    assert "T=4 MAXDIM=16" in text, text
    assert "MAXDIM=64 not 16" in text, text


def test_the_widened_build_selects_the_widened_row_rather_than_refusing():
    """The rule SELECTS as well as rejects, which is why it is worth having.

    At TPU_DMA_WIDEN=1 the live build is DMA_WORDS=16 and the gate must report
    mlp_tiny's 863, not its 1150 and not a failure. Without this, a checker
    that simply refused everything would pass every other test in this file.
    """
    code, text = workload_gate(CLAIMS, build={"TPU_DMA_WIDEN": "1"})
    assert code == 0, text
    assert "DMA_WORDS=16" in text and "863" in text, text
    assert "1150" not in text, text
