"""Dependency-free tests for synthesis artifact collection."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("batch_synthesis.py")
SPEC = importlib.util.spec_from_file_location("batch_synthesis", SCRIPT)
BATCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BATCH
SPEC.loader.exec_module(BATCH)


class ArtifactCollectionTest(unittest.TestCase):
    def test_relative_link_is_copied_and_dangling_link_is_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "worker/results"
            outputs = root / "worker/outputs"
            results.mkdir(parents=True)
            outputs.mkdir()
            (results / "mapped.sdc").write_text("create_clock\n")
            (outputs / "design.sdc").symlink_to("../results/mapped.sdc")
            (outputs / "design.upf").symlink_to("../results/*.upf")

            published = root / "published"
            BATCH.copy_artifact_tree(outputs, published)

            self.assertEqual((published / "design.sdc").read_text(), "create_clock\n")
            self.assertFalse((published / "design.sdc").is_symlink())
            self.assertFalse((published / "design.upf").exists())


if __name__ == "__main__":
    unittest.main()
