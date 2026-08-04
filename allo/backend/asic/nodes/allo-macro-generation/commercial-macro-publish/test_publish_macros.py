"""Tests for hardened-macro publication validation."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("publish_macros.py")
SPEC = importlib.util.spec_from_file_location("publish_macros", SCRIPT)
PUBLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISH
SPEC.loader.exec_module(PUBLISH)


class PublishMacrosTest(unittest.TestCase):
    def test_publishes_intentionally_empty_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs/signoff-batch").mkdir(parents=True)
            (root / "inputs/signoff-batch/index.json").write_text(
                json.dumps({"bypass_macro_generation": True, "entries": []})
            )
            (root / "inputs/asic-manifest-final.json").write_text(
                json.dumps({"macro_groups": []})
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                PUBLISH.main()
            finally:
                os.chdir(previous)
            registry = json.loads(
                (root / "outputs/macro-registry.json").read_text()
            )
            self.assertTrue(registry["bypass_macro_generation"])
            self.assertEqual(registry["implementation_style"], "flat")
            self.assertEqual(registry["macro_count"], 0)
            self.assertEqual(registry["macros"], [])

    def test_accepts_complete_d4_symmetry(self):
        with tempfile.TemporaryDirectory() as temporary:
            lef = Path(temporary) / "macro.lef"
            lef.write_text(
                "MACRO pe\n  CLASS BLOCK ;\n  SYMMETRY X Y R90 ;\nEND pe\n"
            )
            self.assertEqual(PUBLISH.lef_symmetries(lef, "pe"), ["X", "Y", "R90"])

    def test_rejects_missing_rotation_or_wrong_macro(self):
        with tempfile.TemporaryDirectory() as temporary:
            lef = Path(temporary) / "macro.lef"
            lef.write_text("MACRO pe\n  SYMMETRY X Y ;\nEND pe\n")
            with self.assertRaisesRegex(ValueError, "missing symmetries.*R90"):
                PUBLISH.lef_symmetries(lef, "pe")
            with self.assertRaisesRegex(ValueError, "expected macro other"):
                PUBLISH.lef_symmetries(lef, "other")


if __name__ == "__main__":
    unittest.main()
