"""Reuse the SystemC/Catapult PD flow with the small vector-adder source."""

import importlib.util
from pathlib import Path


def construct():
    this_dir = Path(__file__).resolve().parent
    reference = this_dir.parent / "allo-example-systemc" / "construct-commercial.py"
    spec = importlib.util.spec_from_file_location("vadd_systemc_flow", reference)
    reference_flow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_flow)
    graph = reference_flow.construct()
    graph.update_params(
        {
            "construct_path": __file__,
            "allo_design_file": str(this_dir / "allo_design.py"),
            "report_design_name": "vadd-1d",
            "min_macro_reuse": 1,
            "stop_after_step": "none",
        }
    )
    return graph


if __name__ == "__main__":
    construct()
