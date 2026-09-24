import importlib.util
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "worker/scripts/build_coarse_top_layer_obs.py"
)
SPEC = importlib.util.spec_from_file_location("build_coarse_top_layer_obs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CoarseTopLayerObsTests(unittest.TestCase):
    def test_merge_cells_keeps_run_that_reappears_after_row_gap(self):
        cells = {(5, 1), (6, 1), (5, 5), (6, 5)}
        self.assertEqual(
            sorted(MODULE.merge_cells(cells)),
            [(5, 1, 7, 2), (5, 5, 7, 6)],
        )

    def test_hidden_pin_extension_remains_obstructed(self):
        lef = """MACRO sample
  PIN result
    USE SIGNAL ;
    PORT
      LAYER metal5 ;
      RECT 7 5 8 6 ;
    END
  END result
END sample
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.lef").write_text(lef)
            (root / "geometry.tsv").write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t1\n"
                # An earlier identical x-run exercises the row-gap merge case.
                "RECT\twire\tother\t5\t1\t7\t2\n"
                # The real terminal extends inward beyond its edge LEF pin.
                "RECT\twire\tresult\t5\t5\t8\t6\n"
            )
            subprocess.run([
                sys.executable, str(SCRIPT),
                "--input-lef", str(root / "input.lef"),
                "--geometry", str(root / "geometry.tsv"),
                "--output-lef", str(root / "output.lef"),
                "--report", str(root / "report.txt"),
                "--grid-tracks", "1",
                "--pin-grid-tracks", "1",
                "--spacing-tracks", "0",
            ], check=True)
            output = (root / "output.lef").read_text()

        self.assertIn("RECT 5.000000 5.000000 7.000000 6.000000", output)

    @staticmethod
    def sample_lef():
        return """MACRO sample
  PIN result
    PORT
      LAYER metal5 ;
      RECT 2.0 3.0 3.0 4.0 ;
    END
  END result
END sample
"""

    def test_tcl_side_tie_break_expression_is_executable(self):
        result = subprocess.run(
            ["tclsh"],
            input=(
                "set side E\n"
                "set best_side N\n"
                "set side_ratio 0.5\n"
                "set best_ratio 0.5\n"
                "if {$side_ratio < $best_ratio || "
                "($side_ratio == $best_ratio && "
                "[string compare $side $best_side] < 0)} {puts chosen}\n"
            ),
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "chosen")

    def test_macro_pnr_sizes_pin_depth_without_increasing_edge_width(self):
        pnr = (SCRIPT.parent / "pnr.tcl").read_text()
        self.assertIn("-pinWidth $pin_width -pinDepth $pin_depth", pnr)
        self.assertIn(
            "set pin_depth [expr {$pin_width * $depth_width_multiplier}]",
            pnr,
        )
        self.assertIn("ecoRoute -target", pnr)

    def test_merges_cells_and_preserves_pin_access(self):
        lef = """MACRO sample
  PIN result
    PORT
      LAYER metal5 ;
      RECT 0.0 3.0 0.5 3.5 ;
    END
  END result
  OBS
    LAYER metal4 ;
      RECT 0.0 0.0 8.0 8.0 ;
  END
END sample
"""
        self.assertEqual(
            MODULE.top_layer_pin_rectangles(lef, "metal5"),
            [("result", "SIGNAL", (0.0, 3.0, 0.5, 3.5))],
        )
        occupied = {(1, 1), (2, 1), (1, 2), (2, 2)}
        self.assertEqual(MODULE.merge_cells(occupied), [(1, 1, 3, 3)])
        output = MODULE.insert_obs(lef, "metal5", [(1.0, 1.0, 3.0, 3.0)])
        self.assertIn("LAYER metal4", output)
        self.assertIn("LAYER metal5", output)
        self.assertIn("RECT 1.000000 1.000000 3.000000 3.000000", output)

    def test_reads_resolved_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "geometry.tsv"
            path.write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t0.2\n"
                "RECT\twire\t1\t2\t3\t4\n"
            )
            die, layer, pitch, rectangles = MODULE.read_geometry(path)
            self.assertEqual(die, (0.0, 0.0, 8.0, 8.0))
            self.assertEqual(layer, "metal5")
            self.assertEqual(pitch, 0.2)
            self.assertEqual(rectangles, [("wire", "", (1.0, 2.0, 3.0, 4.0))])

    def test_reads_geometry_net_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "geometry.tsv"
            path.write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t0.2\n"
                "RECT\twire\tresult\t1\t2\t3\t4\n"
            )
            _, _, _, rectangles = MODULE.read_geometry(path)
            self.assertEqual(
                rectangles, [("wire", "result", (1.0, 2.0, 3.0, 4.0))]
            )

    def test_normalizes_tcl_quoted_bus_and_hierarchical_names(self):
        self.assertEqual(
            MODULE.normalize_net_name("{v1877399_dout[31]}"),
            "v1877399_dout[31]",
        )
        self.assertEqual(
            MODULE.normalize_net_name(r"{block/signal\[7\]}"),
            "block/signal[7]",
        )

    def test_reader_normalizes_tcl_quoted_net_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "geometry.tsv"
            path.write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t0.2\n"
                "RECT\twire\t{v1877399_dout[31]}\t0\t1\t2\t3\n"
            )
            _, _, _, rectangles = MODULE.read_geometry(path)
            self.assertEqual(
                rectangles,
                [("wire", "v1877399_dout[31]", (0.0, 1.0, 2.0, 3.0))],
            )

    def test_boundary_path_avoids_unrelated_geometry(self):
        path = MODULE.boundary_path({(2, 2)}, {(1, 2), (3, 2)}, 5, 5)
        self.assertTrue(path)
        self.assertEqual(path[0], (2, 2))
        self.assertTrue(path[-1][0] in {0, 4} or path[-1][1] in {0, 4})
        self.assertFalse(set(path) & {(1, 2), (3, 2)})

    def test_rectangle_upper_edge_is_half_open(self):
        cells = set(MODULE.cells_for_rect(
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 4.0, 4.0),
            1.0,
            0.0,
            4,
            4,
        ))
        self.assertEqual(cells, {(0, 0)})

    def test_hybrid_obs_keeps_verified_pin_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.lef").write_text(self.sample_lef())
            (root / "geometry.tsv").write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t1\n"
                "RECT\twire\tresult\t2\t3\t3\t4\n"
                "RECT\twire\tinternal\t6\t6\t7\t7\n"
            )
            subprocess.run([
                sys.executable, str(SCRIPT),
                "--input-lef", str(root / "input.lef"),
                "--geometry", str(root / "geometry.tsv"),
                "--output-lef", str(root / "output.lef"),
                "--report", str(root / "report.txt"),
                "--grid-tracks", "4",
                "--pin-grid-tracks", "1",
                "--spacing-tracks", "0",
            ], check=True)
            report = (root / "report.txt").read_text()
            self.assertIn("pin_grid_tracks 1", report)
            self.assertIn("pin_rectangles 1", report)
            self.assertIn("signal_pin_rectangles 1", report)
            self.assertIn("power_ground_pin_rectangles 0", report)
            self.assertTrue((root / "output.lef").is_file())

    def test_power_ground_stripe_does_not_require_signal_corridor(self):
        lef = """MACRO sample
  PIN VSS
    DIRECTION INOUT ;
    USE GROUND ;
    PORT
      LAYER metal5 ;
      RECT 2.0 3.0 6.0 4.0 ;
    END
  END VSS
END sample
"""
        self.assertEqual(
            MODULE.top_layer_pin_rectangles(lef, "metal5"),
            [("VSS", "GROUND", (2.0, 3.0, 6.0, 4.0))],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.lef").write_text(lef)
            (root / "geometry.tsv").write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t1\n"
                "RECT\tspecial_wire\tVSS\t2\t3\t6\t4\n"
                "RECT\twire\tother\t1\t2\t7\t3\n"
                "RECT\twire\tother\t1\t4\t7\t5\n"
            )
            subprocess.run([
                sys.executable, str(SCRIPT),
                "--input-lef", str(root / "input.lef"),
                "--geometry", str(root / "geometry.tsv"),
                "--output-lef", str(root / "output.lef"),
                "--report", str(root / "report.txt"),
                "--grid-tracks", "4",
                "--pin-grid-tracks", "1",
                "--spacing-tracks", "1",
            ], check=True)
            report = (root / "report.txt").read_text()
            self.assertIn("signal_pin_rectangles 0", report)
            self.assertIn("power_ground_pin_rectangles 1", report)

    def test_power_ground_still_requires_owning_routed_geometry(self):
        lef = """MACRO sample
  PIN VDD
    USE POWER ;
    PORT
      LAYER metal5 ;
      RECT 2.0 3.0 6.0 4.0 ;
    END
  END VDD
END sample
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.lef").write_text(lef)
            (root / "geometry.tsv").write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t1\n"
                "RECT\twire\tother\t2\t3\t6\t4\n"
            )
            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--input-lef", str(root / "input.lef"),
                "--geometry", str(root / "geometry.tsv"),
                "--output-lef", str(root / "output.lef"),
                "--report", str(root / "report.txt"),
                "--grid-tracks", "4",
                "--pin-grid-tracks", "1",
                "--spacing-tracks", "1",
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no matching owning-net geometry", result.stderr)

    def test_hybrid_obs_rejects_unrelated_geometry_in_pin_opening(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.lef").write_text(self.sample_lef())
            (root / "geometry.tsv").write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t1\n"
                "RECT\twire\tresult\t2\t3\t3\t4\n"
                "RECT\twire\tother\t2.5\t3\t3.5\t4\n"
            )
            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--input-lef", str(root / "input.lef"),
                "--geometry", str(root / "geometry.tsv"),
                "--output-lef", str(root / "output.lef"),
                "--report", str(root / "report.txt"),
                "--grid-tracks", "4",
                "--pin-grid-tracks", "1",
                "--spacing-tracks", "0",
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unrelated top-layer geometry", result.stderr)

    def test_hybrid_obs_rejects_pin_without_matching_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.lef").write_text(self.sample_lef())
            (root / "geometry.tsv").write_text(
                "DIE\t0\t0\t8\t8\n"
                "LAYER\tmetal5\n"
                "PITCH\t1\n"
                "RECT\twire\tother\t6\t6\t7\t7\n"
            )
            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--input-lef", str(root / "input.lef"),
                "--geometry", str(root / "geometry.tsv"),
                "--output-lef", str(root / "output.lef"),
                "--report", str(root / "report.txt"),
                "--grid-tracks", "4",
                "--pin-grid-tracks", "1",
                "--spacing-tracks", "0",
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no matching owning-net geometry", result.stderr)


if __name__ == "__main__":
    unittest.main()
